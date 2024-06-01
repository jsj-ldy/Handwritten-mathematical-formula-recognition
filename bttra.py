from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from bttr.utils import Hypothesis

from bttr.model.decoder import Decoder
from bttr.model.encoder import Encoder


class BTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,#模型的维度
        growth_rate: int,#增长率
        num_layers: int,#层数
        nhead: int,#注意力头数
        num_decoder_layers: int,#解码器层数
        dim_feedforward: int,#前馈神经网络的隐藏层维度
        dropout: float,#dropout概率
    ):
        super().__init__()
        #编码器
        self.encoder = Encoder(
            d_model=d_model,#模型的维度
            growth_rate=growth_rate, #增长率
            num_layers=num_layers#层数
        )
        #解码器
        self.decoder = Decoder(
            d_model=d_model,#维度
            nhead=nhead,#注意力头数
            num_decoder_layers=num_decoder_layers,#解码器层数
            dim_feedforward=dim_feedforward,#前馈神经网络的隐藏层维度
            dropout=dropout,#dropout概率
        )


    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:#图像数据，图像掩码，目标数据
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        #编码器将图像数据编码成特征向量
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        #将特征张量feature在维度0上复制一份并连接起来，2b表示批量大小的两倍，t序列长度，d特征向量的维度
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        #也是复制，2b批量大小的两倍，t是掩码的长度
        mask = torch.cat((mask, mask), dim=0)#[2b, t]
        #在维度0上将特征张量feature和掩码张量mask连接起来，扩展为原来的两倍
        #扩展后的特征张量和掩码张量可以用于后续的处理，例如在解码器中进行对齐计算等。
        out = self.decoder(feature, mask, tgt)

        return out

#out 的形状通常是 [batch_size, sequence_length, vocab_size]，
# 其中 batch_size 是批量大小，sequence_length 是目标序列的长度，
# vocab_size 是词汇表的大小，表示每个位置上每个单词的概率分布。

#在给定图像的情况下执行双向 Beam Search（双向束搜索）算法
    def beam_search(
        self, img: FloatTensor, img_mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor输入的图像数据
            [1, 1, h', w']
        img_mask: LongTensor输入图像的掩码
            [1, h', w']
        beam_size : int每个时间步保留的备选结果数量。
        max_len : int生成序列的最大长度限制

        Returns
        -------
        List[Hypothesis]
        """
        #使用编码器对输入的图像数据和掩码进行编码，得到特征张量和掩码张量，t特征序列的长度，d特征向量的维度
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        #该方法将执行双向 Beam Search 算法，并返回一个包含了 Beam Search 结果的列表，
        # 每个元素是一个 Hypothesis 对象，表示一个候选序列及其对应的得分。
        return self.decoder.beam_search(feature, mask, beam_size, max_len)
