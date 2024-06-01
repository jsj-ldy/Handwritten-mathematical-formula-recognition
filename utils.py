from typing import List, Tuple

import editdistance
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric

from bttr.datamodule import vocab

#束搜索算法中生成的假设，即一个候选的翻译结果
class Hypothesis:
    seq: List[int]#一个整数列表，表示假设的序列，每个整数代表一个词的索引
    score: float#假设的得分，得分越高表示假设越优

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()#PyTorch张量（tensor）转换为Python列表（list）的方法。

        if direction == "r2l":
            result = raw_seq[::-1]#对列表进行切片操作，将列表翻转
        else:
            result = raw_seq#原始的列表

        self.seq = result
        self.score = score

    def __len__(self):#得到假设序列的长度
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):#打印假设对象时的字符串表示
        return f"seq: {self.seq}, score: {self.score}"

#PyTorch Lightning的指标（Metric）子类
#计算模型的表达率（Expression Rate）
class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        #参数用于控制指标是否在每个训练步骤同步。在每个epoch结束时同步
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        #两个状态变量：total_line（总行数）和rec（正确识别的行数）
        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")
    #indices_hat和indices，分别表示模型预测的索引序列和真实的索引序列
    def update(self, indices_hat: List[int], indices: List[int]):
        dist = editdistance.eval(indices_hat, indices)
        #它使用编辑距离来计算这两个序列之间的差异
        if dist == 0:
            self.rec += 1#判断正确的个数

        self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate##计算正确率

#交叉熵损失

def ce_loss(
    output_hat: torch.Tensor, output: torch.Tensor, ignore_idx: int = vocab.PAD_IDX
) -> torch.Tensor:
    """
    comput cross-entropy loss
    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):
    Returns:
        torch.Tensor: loss value
    """
    #模型的输出,output_hat展平，batch 是批量大小，len 是序列长度，e 是词嵌入的维度
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    #目标序列，output展平
    flat = rearrange(output, "b l -> (b l)")
    #F.cross_entropy 函数计算交叉熵损失，将 output_hat 中的每个词的预测概率与相应的目标标签进行比较，并计算损失值
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx)
    #函数返回计算得到的损失值
    return loss

#将输入的标记序列转换为目标序列（tgt）和输出序列（out）
def to_tgt_output(
    tokens: List[List[int]], direction: str, device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices
    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}
    #标记序列列表 tokens
    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        #torch.flip 函数，它的作用是沿着指定的维度对张量进行翻转操作
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]
    tgt = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)
    #形成双向的目标序列和输出序列
    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)
    #返回双向的目标序列 tgt 和输出序列 out。
    return tgt, out
