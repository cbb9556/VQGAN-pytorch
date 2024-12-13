import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        # 根据索引获取对应的向量，并调整其形状
        # self.vqgan.codebook.embedding(indices) 从代码本中获取索引对应的嵌入向量
        # .reshape(indices.shape[0], p1, p2, 256) 将获取到的向量调整为指定形状，以适应后续的图像解码过程
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)

        # 调整向量的维度顺序，以符合图像解码的输入要求
        # .permute(0, 3, 1, 2) 将向量的维度从 (batch_size, p1, p2, 256) 调整为 (batch_size, 256, p1, p2)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)

        # 解码向量为图像
        # self.vqgan.decode(ix_to_vectors) 使用VQ-GAN的解码器将调整好形状和维度顺序的向量解码为图像
        image = self.vqgan.decode(ix_to_vectors)

        # 返回解码后的图像
        return image

    def forward(self, x):
        _, indices = self.encode_to_z(x)

        # 生成SOS标记，形状与输入批次大小匹配 self.sos_token= 0
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        # 确保SOS标记为长整型，并移至CUDA设备以利用GPU加速
        sos_tokens = sos_tokens.long().to("cuda")

        # 生成保留概率掩码，用于决定是否保留原始索引
        # 假设 indices 的形状为 (2, 5)，则 torch.ones(indices.shape, device=indices.device) 生成一个形状为 (2, 5) 的全1张量。
        # self.pkeep 为 0.8，所以 self.pkeep * torch.ones(indices.shape, device=indices.device) 生成一个形状为 (2, 5) 的张量，每个元素都是 0.8。
        # torch.bernoulli 函数根据每个元素的self.pkeep = 0.5 概率生成一个二进制掩码。
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        # 将掩码四舍五入，确保其为整型
        mask = mask.round().to(dtype=torch.int64)
        # 生成随机索引，用于替换原始索引
        # 假设 indices 的形状为 (2, 5)，则 torch.randint_like(indices, self.transformer.config.vocab_size=1000)
        # 生成一个形状为 (2, 5) 的张量，每个元素是从 0 到 999 之间的随机整数。
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        # 根据掩码决定使用原始索引还是随机索引
        # mask 中为 1 的位置保留 indices 中的值，为 0 的位置使用 random_indices 中的值。
        new_indices = mask * indices + (1 - mask) * random_indices

        # 在新索引前添加SOS标记，为后续处理做准备
        #将 sos_tokens 沿着第1列 拼接到 new_indices 的前面。
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        # 预测mask住的位置的 codebook的 indices值的 概率分布
        # 具体来说，new_indices[:, :-1] 表示取每个序列的前 (seq_length - 1) 个元素作为输入，也就是最后1列的数据不用输入
        # sos-token就是起始token
        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        # 将transformer模型设置为评估模式，以禁用dropout等仅在训练时需要的特性
        self.transformer.eval()

        # 将条件张量c与输入张量x沿维度1（列方向）拼接，以适应模型输入要求
        x = torch.cat((c, x), dim=1)

        # 按步骤数循环，逐步生成序列，256步，生成256个token，从sos-token开始
        for k in range(steps):
            # 通过transformer模型前向传播，获取未归一化的预测分布（logits）
            logits, _ = self.transformer(x)
            # 仅保留最后一个时间步的预测，并应用温度参数以调整预测的不确定性，温度越高， 词汇表的概率分布越均匀，越能采样出多样的结果
            logits = logits[:, -1, :] / temperature

            # 如果设置了top_k参数，则只保留每个位置概率最大的top_k个标记的logits，以增加生成文本的多样性
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            # 对logits进行softmax激活，转换为概率分布
            probs = F.softmax(logits, dim=-1)

            # 根据概率分布进行多项式采样，以决定下一个时间步的输入
            # 多项式采样: 从给定的概率分布中随机抽取样本。probs 中的每个元素表示对应索引位置的元素被选中的概率。
            # num_samples=1: 表示从概率分布中抽取一个样本。返回的结果是一个包含单个索引的张量。
            ix = torch.multinomial(probs, num_samples=1)

            # 将采样的标记拼接到输入序列x中，以进行下一个时间步的预测
            x = torch.cat((x, ix), dim=1)

        # 移除输出序列中的条件部分，仅保留生成的部分
        x = x[:, c.shape[1]:]

        # 将transformer模型恢复为训练模式
        self.transformer.train()

        # 返回生成的序列
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))
















