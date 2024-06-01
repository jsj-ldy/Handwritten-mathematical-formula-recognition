import math
from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat

#词级别的位置编码器，为输入序列中的每个词向量添加位置信息。
#位置编码就被融合到了输入特征张量中，
#使得模型可以利用位置信息进行更加准确的学习。
class WordPosEnc(pl.LightningModule):
    def __init__(
        self, d_model: int = 512, max_len: int = 500, temperature: float = 10000.0
    ) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)#存储位置编码。
        #序列表示了位置编码张量中的位置索引。
        position = torch.arange(0, max_len, dtype=torch.float)
        #位置编码中不同维度位置的索引，偶数位置对应着正弦函数的值，而奇数位置对应着余弦函数的值。
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))
        #维度索引除以模型的维度，然后将结果求幂，得到一个新的张量
        #temperature 的值以 dim_t / d_model 为指数进行求幂的操作
        #这个操作通常用于调整位置编码中不同维度位置的频率。
        #torch.einsum 函数执行了一个张量乘积的操作
        inv_freq = torch.einsum("i, j -> i j", position, div_term)
        #乘积的结果是一个形状为 (max_len, d_model / 2) 的张量，
        #其中每一行是 position 张量的相应元素乘以 div_term 张量的所有元素的乘积。
        pe[:, 0::2] = inv_freq.sin()#所有行的偶数索引列
        pe[:, 1::2] = inv_freq.cos()#所有行的奇数索引列
        self.register_buffer("pe", pe)
        #将名为 "pe" 的缓冲张量注册到了当前的 Module 中
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        #其中 b 是批量大小，l 是序列长度，d 是特征的维度
        _, seq_len, _ = x.size()
        #pe 的形状是 [max_len, d_model]，
        #其中 max_len 是最大序列长度，
        # 所以需要取出 pe 中与当前序列长度相对应的部分，
        emb = self.pe[:seq_len, :]
        #将其扩展为与输入特征张量相同的形状
        x = x + emb[None, :, :]
        return x

#图像的位置编码（Positional Encoding），用于将位置信息融合到图像特征张量中
class ImgPosEnc(pl.LightningModule):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000.0,#温度参数，用于计算位置编码中的频率参数
        normalize: bool = False,#如果 normalize 设置为 True，则需要提供一个缩放因子，用于对位置编码进行缩放。
        scale: Optional[float] = None,#归一化时的缩放因子
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale#缩放因子的范围

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """add image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask#将输入的布尔张量中的每个元素取反
        y_embed = not_mask.cumsum(1, dtype=torch.float32)#沿着第二个维度（即图像的高度方向）进行累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32)#沿着第三个维度（即图像的宽度方向）进行累加。
        if self.normalize:#位置编码归一化处理
            eps = 1e-6
            #它通过除以最后一个位置的累加位置编码值，然后乘以一个缩放因子 scale 来进行归一化。
            #归一化后的位置编码使得在不同图像尺寸下，位置编码的范围保持一致，有利于模型的训练。
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # not exactly the same as concat two WordPosEnc
        # WordPosEnc: sin(0), cos(0), sin(2), cos(2)
        # ImagePosEnc: sin(0), cos(1), sin(2), cos(3)
        dim_t = torch.arange(self.half_d_model, dtype=torch.float, device=self.device)
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))
        #self.temperature 的值分别除以
        # dim_t 中每个位置的值除以 self.half_d_model 的结果的幂
        #torch.einsum 函数用于计算张量的乘积
        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)
        #pos_x 张量沿着第四维度（索引从0开始）切片，
        # 然后对切片后的张量进行正弦和余弦的计算，最后stack将这两个结果沿着第四维度进行堆叠。
        #将堆叠后的张量在第四维度上展平为三维张量。
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        #（B，H，W，2，D/2）->（B，H，W，2*D/2）
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        #pos_x 和 pos_y 按照最后一个维度进行连接，得到最终的位置编码张量 pos。
        pos = torch.cat((pos_x, pos_y), dim=3)

        x = x + pos
        return x
#每两个元素 x1 和 x2 替换为 (-x2, x1)
#将输入张量中的相邻两个元素进行旋转，
def rotate_every_two(x: torch.FloatTensor):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)#沿着最后一个维度进行拆分,x1偶数索引处的元素，x2奇数索引处的元素
    x = torch.stack((-x2, x1), dim=-1)#就是把最后的奇数索引处的元素取反，偶数索引处的元素与其进行互换
    return rearrange(x, "... d j -> ... (d j)")

#自注意力模型中用于添加位置编码的方法
#优点在于它能够避免在位置编码中使用三角函数，
# 而是通过矩阵乘法和张量操作来实现位置编码的添加。
#为输入的序列数据添加旋转位置编码
class WordRotaryEmbed(pl.LightningModule):
    """
    Rotary Positional Embedding
    Ref : https://zhuanlan.zhihu.com/p/359502624
        : https://blog.eleuther.ai/rotary-embeddings/
        : https://arxiv.org/abs/2104.09864

    lucidrains implementation: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/rotary.py
    """

    def __init__(self, d_model: int = 512, temperature: float = 10000.0) -> None:
        super().__init__()
        #这部分代码计算了逆频率张量，
        inv_freq = 1.0 / (
            #这部分代码将频率张量除以d_model，然后将其除以temperature的幂
            #为了调整频率张量的值，使得它们的取值范围在一个合适的范围内，
            temperature ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        #将张量 inv_freq 注册为模型的缓冲区，
        #注册的缓冲区在模型进行前向传播时不会被更新，
        #也不会被包含在模型的可学习参数中，但是可以被模型访问和使用。
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.FloatTensor):
        """apply positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        _, n, _ = x.size()
        #包含了从 0 到 n-1 的整数序列，并且张量的数据类型与 self.inv_freq 张量相匹配
        #位置索引张量
        t = torch.arange(n, device=self.device).type_as(self.inv_freq)
        #逆频率张量与位置索引张量进行点积运算
        sinusoid_inp = torch.einsum("i, j -> i j", t, self.inv_freq)#(n, d_model/2)
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        #repeat 函数沿着第二个维度 n 进行重复，并且在结果张量的最后一维度上将其重复两次。
        #在最后一维度上的维度数量变成了原来的两倍。
        #对正弦和余弦值进行适当的重复操作以与输入特征张量的维度相匹配，
        sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))
        #余弦位置编码+正弦位置编码,有效地将位置信息融入到输入张量中
        #旋转操作是为了确保正弦和余弦位置编码在特征维度上交替出现。
        # 这是因为正弦和余弦函数的周期性质，使得它们在特征维度上重复出现。
        # 通过旋转操作，可以保证正弦和余弦位置编码均匀地分布在特征维度上，
        # 从而更好地捕捉到不同位置的信息。
        #rotate_every_two 函数会将输入张量的特征维度中的相邻两个元素进行旋转
        x = (x * cos) + (rotate_every_two(x) * sin)
        return x

#为输入的图像增加旋转位置编码
class ImageRotaryEmbed(pl.LightningModule):
    """
    2-D Generalized version of WordRotaryEmbedding
    """

    def __init__(
        self,
        d_model: int = 512,#特征的维度
        temperature: float = 10000,#温度参数
        normalize: bool = False,#是否归一化
        scale: Optional[float] = None,#位置编码的缩放因子，默认为None，如果不为None，则必须设置normalize为True。
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    #默认的缩放因子是2π，这是一个常用的选择，
    #因为它使得位置编码的取值范围在一个合适的区间内，
    #并且可以与三角函数相结合使用，以便将位置编码添加到特征张量中。



    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """apply image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        #计算位置索引张量
        not_mask = ~mask#反掩码，掩码区域转换为0，非掩码区域转换为1
        #沿着图像的高度，对反掩码进行累积求和，位置索引张量
        embed_y = not_mask.cumsum(1, dtype=torch.float32)
        #宽度方向对反掩码进行累积求和
        embed_x = not_mask.cumsum(2, dtype=torch.float32)
        #归一化位置索引
        if self.normalize:
            eps = 1e-6#使用了一个小的正数eps来避免除以零的情况。
            #每个位置索引值都被除以最后一个位置的累积和，这样可以将所有位置的值缩放到0到1之间
            #归一化的位置索引张量被乘以一个缩放因子，确保位置编码的值处于一个合理的范围内，比如常用的范围是2π。
            embed_y = embed_y / (embed_y[:, -1:, :] + eps) * self.scale
            embed_x = embed_x / (embed_x[:, :, -1:] + eps) * self.scale
        #这个序列从0开始，以2为步长，直到self.half_d_model（即特征维度的一半）。
        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        #除以频率因子的幂来计算逆频率张量
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))
        # 将归一化的位置索引张量与逆频率张量相乘
        # [b, h, w, d_model // 4]
        pos_x = torch.einsum("b h w, d -> b h w d", embed_x, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", embed_y, inv_feq)
        #生成正弦和余弦编码
        # [b, h, w, d_model // 2]
        sin_x, cos_x, sin_y, cos_y = map(
            lambda t: repeat(t, "b h w d -> b h w (d n)", n=2),
            (pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()),
        )
        # [b, h, w, d_model]
        #张量和起来
        sin = torch.cat((sin_x, sin_y), dim=-1)
        cos = torch.cat((cos_x, cos_y), dim=-1)
        #将图像中每个像素点的位置信息编码到特征张量中，以增强模型对图像内容的理解。
        x = (x * cos) + (rotate_every_two(x) * sin)
        return x
