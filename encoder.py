import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from bttr.model.pos_enc import ImageRotaryEmbed, ImgPosEnc


# DenseNet-B
#作用，减少特征图的维度，增加特征图的宽度
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        #super(_Bottleneck, self) 返回了 _Bottleneck 类的父类对象，调用了该对象的init方法
        #super() 函数返回一个代理对象，通过它可以调用父类的方法。
        interChannels = 4 * growth_rate
        #计算了瓶颈层中间通道的数量，中间通道的数量被定义为增长率的四倍
        #增长率：在稠密块中，每个卷积层输出的特征图的通道数，通过将中间通道的数量设置为增长率的四倍
        #也就是在每个瓶颈层中增加图的维度
        # 归一化是指将数据按比例缩放，使其落入特定范围
        # 二位批量归一化层，标准化输入特征图沿批次和空间维度的数据分布，可以加速收敛和提高稳定性，有正则化的效果
        self.bn1 = nn.BatchNorm2d(interChannels)
        #二维卷积层，作用是将输入特征图通过一个1x1的卷积核进行卷积操作，并将通道数从 n_channels 调整为 interChannels。
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        #归一化
        self.bn2 = nn.BatchNorm2d(growth_rate)
        #二维卷积层
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)
    #瓶颈层的前向传播
    def forward(self, x):
        #先1*1卷积再归一化，再用relu激活函数处理
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        #对第一次处理后的特征图进行操作
        if self.use_dropout:
            out = self.dropout(out)
        #再经过3*3的卷积层和规一化，再relu进行处理
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        #dropout操作
        out = torch.cat((x, out), 1)
        #将输入特征图和处理后的特征图拼接在通道维度1上
        return out#输出

#单层的密集连接层，作用是将输入的特征图与经过卷积和批量归一化处理的特征图进行连接。
# single layer
class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)
#forward是真正实现操作的地方
    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
           #输入特征图 x 和处理后的特征图拼接在通道维度上（dim=1），
            #然后作为单层的输出返回。
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        #平均池化减小特征图的尺寸
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,#增长率，输出通道数
        num_layers: int,#层数
        reduction: float = 0.5,#在过渡层中降低特征图的维度的比例
        bottleneck: bool = True,#表示是否在稠密块中使用瓶颈结构
        use_dropout: bool = True,#dropout 是一种正则化技术，有助于防止过拟合。
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate#通道数
        self.conv1 = nn.Conv2d(
            1, n_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(n_channels)#归一化
        #稠密块，包含多个稠密层的序列，n_channels输入通道数，growth_rate 每个稠密块中的通道数增长率，
        #n_dense_blocks: DenseNet 中稠密块的数量。
        self.dense1 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        #每个稠密块的输出通道数等于当前通道数加上稠密块中每个稠密层的通道数增长率乘以稠密块的层数。
        #确保与前一个稠密块的输出通道数相匹配
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))#一一定的比例减少通道数
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    #创建密集块
    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        #创建指定数量 n_dense_blocks 的密集块
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        #在创建每一层时，输入通道数 n_channels 会不断增加，以适应下一层的输入。
        return nn.Sequential(*layers)
        #如 _Bottleneck 或 _SingleLayer，然后将这些层组合成一个 nn.Sequential 容器返回。

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]#下采样是指通过减少输入数据的空间维度来降低其分辨率。对掩码进行下采样，以匹配池化后的输出大小。
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)#执行最大池化操作
        out_mask = out_mask[:, 0::2, 0::2]#对下采样后的掩码进行更新
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]#更新
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]#更新
        out = self.dense3(out)
        out = self.post_norm(out)#归一化
        return out, out_mask#返回输出和对应的掩码


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)
        #有效地将 DenseNet 提取的特征图转换为指定维度的特征表示
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.model.out_channels, d_model, kernel_size=1),
            nn.ReLU(inplace=True),#激活函数增加非线性
        )
        #归一化
        self.norm = nn.LayerNorm(d_model)
        #图像位置编码
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, t, d], [b, t]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)#使用 DenseNet 模型对输入的图像进行特征提取
        feature = self.feature_proj(feature)#进行通道数的转换和非线性变换

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")#对特征张量进行维度重排
        feature = self.norm(feature)#归一化

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)#对特征张量应用二维位置编码

        # flat to 1-D
        #将特征张量和掩码张量展平为二维
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")
        print(feature)
        print(mask)
        return feature, mask
