import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from bttr.datamodule import Batch, vocab
from bttr.model.bttra import BTTR
from bttr.utils import ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out


class LitBTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int=512,#模型的维度
        # encoder
        growth_rate: int=32,#增长率
        num_layers: int=3,#层数
        # decoder
        nhead: int=4,#注意力头数
        num_decoder_layers: int=3,#解码器层数
        dim_feedforward: int=2048,#前馈神经网络的隐藏层维度
        dropout: float=0.1,#dropout概率
        # beam search
        beam_size: int=10,#束搜索的大小
        max_len: int=200,#最大序列长度
        alpha: float=1.0,#束搜索中长度惩罚的参数
        # training
        learning_rate: float=1e-3,#学习率
        patience: int=5,#EarlyStopping的耐心参数，指示在停止训练之前等待多少个epoch
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bttr = BTTR(
            d_model=d_model,#模型的维度
            growth_rate=growth_rate,#增长率
            num_layers=num_layers,#层数
            nhead=nhead,#注意力头数
            num_decoder_layers=num_decoder_layers,#解码器层数
            dim_feedforward=dim_feedforward,#前馈神经网络的隐藏层维度
            dropout=dropout,#dropout概率
        )
        #模型的成功率
        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
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
        #它调用了内部的 BTTR 模型，将图像和目标序列作为输入，并返回预测的目标序列,
        #返回两个，一个是这个输出表示每个位置上每个单词的概率分布。一个是用于生成给定图像的文本描述。
        return self.bttr(img, img_mask, tgt)

    #推断阶段的束搜索，它接受一个图像张量作为输入，并返回一个最佳的目标序列的 LaTex 字符串。
    def beam_search(
        self,
        img: FloatTensor,#图像张量
        beam_size: int = 1,#可选参数
        max_len: int = 200,#可选参数
        alpha: float = 1.0,#可选参数
    ) -> str:
        """for inference, one image at a time

        Parameters
        ----------
        img : FloatTensor
            [1, h, w]
        beam_size : int, optional
            by default 10
        max_len : int, optional
            by default 200
        alpha : float, optional
            by default 1.0

        Returns
        -------
        str
            LaTex string
        """
        assert img.dim() == 3
        #创建一个与输入图像相同形状的图像掩码 img_mask
        img_mask = torch.zeros_like(img, dtype=torch.long)  # squeeze channel
        #获取候选序列的列表 hyps
        hyps = self.bttr.beam_search(img.unsqueeze(0), img_mask, beam_size, max_len)
        #选择最佳的候选序列，每个候选序列的得分除以其长度的指数倍，从而对候选序列进行惩罚，每个候选序列的得分通常会随着序列的长度增加而增加
        #将每个候选序列的得分除以其长度的指数倍。
        #假设一个候选序列的长度为l，得分为s，则对该候选序列进行长度惩罚后的得分为s/l的α次幂
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
       #返回最佳候选序列对应的文本描述。
        #最终选择的候选序列可能更倾向于短一些的序列，而不是简单地选择得分最高的序列。
        #这有助于减轻长序列的偏向，使得生成的序列更加合理和平衡。
        return vocab.indices2label(best_hyp.seq)
    #用于计算模型的损失并记录训练日志。
    def training_step(self, batch: Batch, _):
        ##返回双向的目标序列 tgt 和输出序列 out。
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        #目标序列将用作模型的输入和期望输出。
        out_hat = self(batch.imgs, batch.mask, tgt)
        #计算预测输出和实际输出之间的交叉熵损失
        loss = ce_loss(out_hat, out)
        #将计算得到的损失值记录到训练日志中，以便后续的训练过程中进行监控和可视化。
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        #返回计算得到的损失值，以便于在训练过程中进行优化和反向传播。
        return loss

    def validation_step(self, batch: Batch, _):
        #获取模型输出和计算损失
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)
        loss = ce_loss(out_hat, out)
        #记录验证损失
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,#示该指标在每个 epoch 结束时被记录
            prog_bar=True,#表示该指标将显示在进度条中
            sync_dist=True,
        )
        #执行 Beam Search
        hyps = self.bttr.beam_search(
            batch.imgs, batch.mask, self.hparams.beam_size, self.hparams.max_len
        )
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))
        #记录正确率（准确率）
        self.exprate_recorder(best_hyp.seq, batch.indices[0])
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.bttr.beam_search(
            batch.imgs, batch.mask, self.hparams.beam_size, self.hparams.max_len
        )
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))
        self.exprate_recorder(best_hyp.seq, batch.indices[0])
        #你将获得测试图像及其对应的模型生成的最佳候选序列
        return batch.img_bases[0], vocab.indices2label(best_hyp.seq)
    #你正在处理测试集的输出并计算最终的表现度量（如成功率）
    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()#计算模型的成功率
        print(f"ExpRate: {exprate}")#打印成功率

        print(f"length of total file: {len(test_outputs)}")#测试输出文件的总数
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_base, pred in test_outputs:
                #%{img_base} 插入了图像基础信息，${pred}$ 插入了模型预测结果
                #.encode() 方法将字符串编码为字节格式，以便写入到文件中。
                content = f"%{img_base}\n${pred}$".encode()
                with zip_f.open(f"{img_base}.txt", "w") as f:
                    f.write(content)
#将测试输出保存到名为 "result.zip" 的 ZIP 文件中，每个文件包含图像基础信息和模型预测结果。

    def configure_optimizers(self):
        #优化器
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,#学习率
            eps=1e-6,#参数 eps
            weight_decay=1e-4,#权重衰减
        )
        #学习率调度器
        #当监测指标 "val_ExpRate" 不再提升时，学习率将按照给定的因子 factor （这里是减小学习率，因此是降低）进行调整。
        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,#需要调度学习率的优化器。
            mode="max",#监测指标的模式，表示当监测指标达到最大值时学习率将被调整。
            factor=0.1,#用于降低学习率的因子，每次调整时，学习率将乘以该因子
            #用于指示在验证指标没有改善时等待多少个检查点(epoch)。
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )#表示在验证指标不再改善时等待的 epoch 数。check_val_every_n_epoch，它表示每隔多少个 epoch 就会进行一次验证。
        scheduler = {
            "scheduler": reduce_scheduler,#指定要使用的学习率调度器对象
            "monitor": "val_ExpRate",#指定要监视的指标，这里是 "val_ExpRate"，表示要监视验证集上的正确率（ExpRate）
            "interval": "epoch",#指定调度器在何时进行调度。
            "frequency": self.trainer.check_val_every_n_epoch,#指定调度器调度的频率
            "strict": True,#指定是否启用严格模式，在严格模式下，如果监视的指标没有改善，学习率将按照给定的调度规则进行调整
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}