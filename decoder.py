from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor
from torch.nn.modules.transformer import TransformerDecoder

from bttr.datamodule import vocab, vocab_size
from bttr.model.pos_enc import WordPosEnc, WordRotaryEmbed
from bttr.utils import Hypothesis, to_tgt_output


def _build_transformer_decoder(
    d_model: int,#维度
    nhead: int,#注意力头数量
    num_decoder_layers: int,#解码器层数
    dim_feedforward: int,#前馈网络的隐藏层维度
    dropout: float,#控制模型中随机失活（Dropout）操作的参数。
) -> nn.TransformerDecoder:
    """build transformer decoder with params
    Parameters
    ----------
    d_model : int
    nhead : int
    num_decoder_layers : int
    dim_feedforward : int
    dropout : float
    Returns
    -------
    nn.TransformerDecoder
    """
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model,#维度
        nhead=nhead,#注意力头数
        dim_feedforward=dim_feedforward,#前馈网络的隐藏层维度
        dropout=dropout,#控制模型中的随机失活操作的参数
    )

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    return decoder
#构建好的解码器对象
class Decoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,#模型维度
        nhead: int,#注意力头数
        num_decoder_layers: int,#解码器层数
        dim_feedforward: int,#前馈网络的隐藏层维度
        dropout: float,#制模型中的随机失活操作的参数
    ):
        super().__init__()
        #词嵌入层
        #nn.Embedding 层将输入的词索引映射为对应的词向量。
        #nn.Sequential 容器来依次应用词嵌入层和 LayerNorm 层对词向量进行处理
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )
        #位置编码
        self.pos_enc = WordPosEnc(d_model=d_model)
        #transformer解码器
        self.model = _build_transformer_decoder(
            d_model=d_model,#模型维度
            nhead=nhead,#注意力头数
            num_decoder_layers=num_decoder_layers,#解码器层数
            dim_feedforward=dim_feedforward,#前馈网络的隐藏层维度
            dropout=dropout,#控制模型失活的参数
        )
        #线性层（nn.Linear）将解码器的输出特征映射到词汇表大小的空间
        self.proj = nn.Linear(d_model, vocab_size)

    #注意力掩码，为了防止模型看到当前时刻之后的标记，需要将未来时刻的信息屏蔽掉。
    #自回归（autoregressive）的注意力掩码，因果（causal）注意力掩码
    #掩盖了当前位置之后的所有位置，因此模型在生成每个位置的输出时只能依赖于已生成的标记
    def _build_attention_mask(self, length):#序列的长度
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )#表示张量大小的元组，表示每个位置之间都存在注意力关系
        mask.triu_(1)  # zero out the lower diagonal
        #triu_ 方法将掩码的下三角区域（包括对角线）全部置零，
        #以实现只关注当前位置之前的信息，而不考虑当前位置之后的信息。
        return mask
#返回构建好的注意力掩码张量，保证在后续的注意力计算中会被使用，
#以确保模型在解码时只能依赖已生成的标记

#前向传播方法，给定源序列和目标序列，生成目标序列的输出
    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, t, d]
        src_mask: LongTensor
            [b, t]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        #生成目标序列的注意力掩码
        tgt_mask = self._build_attention_mask(l)
        #生成目标序列的填充掩码，对应于词汇表中的填充符号 <pad> 的位置
        tgt_pad_mask = tgt == vocab.PAD_IDX
        #词嵌入，目标序列 tgt 通过词嵌入层 word_embed 映射为词向量
        tgt = self.word_embed(tgt)  # [b, l, d]
        #位置编码，pos_enc 对目标序列中的词向量进行位置编码，以捕获每个词的位置信息。
        tgt = self.pos_enc(tgt)  # [b, l, d]
        #输入张量，调整为transformer模型所需的形状
        src = rearrange(src, "b t d -> t b d")
        #目标张量，调整为transformer模型所需的形状
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,#调整后的目标张量
            memory=src,#同时传入源序列
            tgt_mask=tgt_mask,#目标序列的注意力掩码
            tgt_key_padding_mask=tgt_pad_mask,#目标序列的填充掩码
            memory_key_padding_mask=src_mask,#源序列的填充掩码
        )
        #Transformer解码器的输出通过线性投影层
        # proj 映射为目标序列的预测结果
        out = rearrange(out, "l b d -> b l d")
        #(sequence_length, batch_size, embedding_size)
        out = self.proj(out)
        #(batch_size, sequence_length, vocab_size)，
        # 表示每个位置上词汇表中每个词的预测概率分布。
        return out
    #执行束搜索（beam search）算法，以生成最佳的翻译结果。
    # 束搜索是一种启发式搜索算法，常用于生成序列数据，
    # 如机器翻译、语音识别等任务中。
    def _beam_search(
        self,
        src: FloatTensor,#源序列
        mask: LongTensor,#掩码序列
        direction: str,#确定起始和停止标记
        beam_size: int,#束宽
        max_len: int,
    ) -> List[Hypothesis]:#束搜索算法中生成的假设，即一个候选的翻译结果
        """run beam search for one direction

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        direction : str
            one of "l2r" and "r2l"
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        assert direction in {"l2r", "r2l"}
        assert (
            src.size(0) == 1 and mask.size(0) == 1
        ), f"beam search should only have single source, encounter with batch_size: {src.size(0)}"

        if direction == "l2r":
            start_w = vocab.SOS_IDX
            stop_w = vocab.EOS_IDX
        else:
            start_w = vocab.EOS_IDX
            stop_w = vocab.SOS_IDX
        #初始化假设矩阵 hypotheses
        hypotheses = torch.full(
            (1, max_len + 1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=self.device,
        )
        hypotheses[:, 0] = start_w
        #初始化假设得分矩阵 hyp_scores，用于记录每个假设的得分
        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)
        completed_hypotheses: List[Hypothesis] = []

        t = 0
        #在循环中，模型根据当前的假设，源序列，掩码生成下一时刻的预测输出，并计算对应的得分
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)#当前的假设数量
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"
            #复制张量，适应当前的假设数量
            #squeeze(0) 操作将 src 张量的第一个维度（通常是 batch 维度）压缩，
            # 将大小为 1 的维度移除，从而得到一个形状为 (t, d) 的张量，
            # 其中 t 表示时间步数，d 表示特征维度。
            #利用 repeat 函数对张量进行复制。参数 "s e -> b s e" 指定了复制的规则，
            # 表示将输入的 (s, e) 形状的张量复制为 (b, s, e) 形状的张量，
            # 其中 b 是复制的次数（即 hyp_num）。
            #适配当前的假设数量，repeat对维度调整，匹配模型输入的要求
            #准备数据
            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
            exp_mask = repeat(mask.squeeze(0), "s -> b s", b=hyp_num)
            #生成预测输出， 调用模型的前向传播方法（self()），
            # 传入复制后的源序列、掩码以及当前假设，生成预测输出
            decode_outputs = self(exp_src, exp_mask, hypotheses)[:, t, :]
            #通过 softmax 函数计算预测输出的对数概率。
            log_p_t = F.log_softmax(decode_outputs, dim=-1)
            #选择出最优的 live_hyp_num 个假设作为下一时刻的候选假设
            live_hyp_num = beam_size - len(completed_hypotheses)
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e=vocab_size)
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            #这里使用 torch.topk 函数进行选择，并将结果进行解码，
            # 得到新的假设及其得分。
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                continuous_hyp_scores, k=live_hyp_num
            )
            #将当前的候选假设的位置和词汇id进行解码，更新假设列表中的内容
            #用于确定每个候选假设在前一个时间步的位置
            prev_hyp_ids = top_cand_hyp_pos // vocab_size
            #用于确定每个候选假设当前时间步的词汇 ID
            hyp_word_ids = top_cand_hyp_pos % vocab_size

            t += 1
            new_hypotheses = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                #为了将张量的值作为普通的 Python 数值进行处理
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                #更新了每个候选假设当前时间步的生成词汇。
                hypotheses[prev_hyp_id, t] = hyp_word_id
                #如果等于停止词汇，则表示当前假设已经生成了完整的序列，
                #因此将其添加到已完成假设列表 completed_hypotheses 中
                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(
                            #目的是生成一个新的张量，该张量与原始张量具有相同的数值，
                            #但不再与计算图关联，即不会跟踪梯度信息，同时也避免了原始张量被修改。
                            seq_tensor=hypotheses[prev_hyp_id, 1:t]
                            .detach()
                            .clone(),  # remove START_W at first
                            score=cand_new_hyp_score,
                            direction=direction,
                        )
                    )
                else:
                    #如果当前时间步生成的词汇不是停止词汇，当前假设更新到新的假设列表中
                    new_hypotheses.append(hypotheses[prev_hyp_id].detach().clone())
                    #将对应的分数添加到新的分数列表中
                    new_hyp_scores.append(cand_new_hyp_score)
            #完成当前的假设
            if len(completed_hypotheses) == beam_size:
                break
            #将这批张量沿着维度 0 进行堆叠，即将它们按顺序堆叠在一起，形成一个新的张量。
            #将更新后的假设列表转换为一个张量
            hypotheses = torch.stack(new_hypotheses, dim=0)
            #(seq_len, vocab_size)变为(batch_size, seq_len, vocab_size)
         #最后，将已完成的假设添加到结果列表中，并返回束搜索的最终结果。
        if len(completed_hypotheses) == 0:
        #如果没有任何假设被完成（即completed_hypotheses为空），
        #则会将第一个假设添加到completed_hypotheses列表中。
            completed_hypotheses.append(
                Hypothesis(
                    #选择了第一个假设的生成序列，从序列中的第二个位置（索引为1）开始到末尾，
                    #即去除了序列开头的起始词汇
                    seq_tensor=hypotheses[0, 1:].detach().clone(),#确保生成的对象不会与原始张量共享内存，并且可以独立使用
                    score=hyp_scores[0].detach().item(),#获取分数
                    direction=direction,#方向属性
                )
            )

        return completed_hypotheses#返回束搜索的最终结果，就是生成多个假设序列
#方法用于计算给定一批假设的交叉熵损失，并将损失值作为得分添加到每个假设中
#这个方法的作用是评估另一个模型生成的输出与当前模型生成的输出之间的差异，
    # 以此来调整或改进当前模型的生成结果。
    def _cross_rate_score(
        self,
        src: FloatTensor,#源张量
        mask: LongTensor,#掩码张量
        hypotheses: List[Hypothesis],#假设
        direction: str,#方向
    ) -> None:
        """give hypotheses to another model, add score to hypotheses inplace

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask : LongTensor
            [1, l]
        hypotheses : List[Hypothesis]
        direction : str
        """
        assert direction in {"l2r", "r2l"}
        indices = [h.seq for h in hypotheses]
        #每个假设，提取其生成的序列，并将这些序列组合成一个张量 tgt，表示目标序列。
        tgt, output = to_tgt_output(indices, direction, self.device)
        #将假设的生成序列转换为相应的输出张量 output，以便与模型生成的输出进行比较
        b = tgt.size(0)
        #扩展与目标张量的批量大小匹配
        exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
        exp_mask = repeat(mask.squeeze(0), "s -> b s", b=b)
        #获取模型生成的输出张量 output_hat。
        output_hat = self(exp_src, exp_mask, tgt)
        #进行形状变换，rearrange重新排列张量的维度
        flat_hat = rearrange(output_hat, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        #然后使用 PyTorch 的交叉熵损失函数 F.cross_entropy 计算损失值。
        loss = F.cross_entropy(
            flat_hat, flat, ignore_index=vocab.PAD_IDX, reduction="none"
        )

        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)

        for i, l in enumerate(loss):
            score = -l#遍历了损失张量 loss 中的每个样本的损失值 l，
            # 并将其作为负数赋值给对应假设的得分 score
            #。得分的计算采用了负损失值，这是因为通常情况下，
            # 分数越高表示效果越好，而交叉熵损失是一个越小越好的指标
            hypotheses[i].score += score

    def beam_search(
        self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run beam search for src img

        Parameters
        ----------
        src : FloatTensor
            [1, l, d]
        mask: LongTensor
            [1, l]
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        #得到假设列表
        l2r_hypos = self._beam_search(src, mask, "l2r", beam_size, max_len)
        #对假设列表进行打分
        self._cross_rate_score(src, mask, l2r_hypos, direction="r2l")

        r2l_hypos = self._beam_search(src, mask, "r2l", beam_size, max_len)
        self._cross_rate_score(src, mask, r2l_hypos, direction="l2r")
        return l2r_hypos + r2l_hypos
        #这种双向束搜索的思想是为了更全面地探索可能的解空间，
        #从而提高模型生成的结果的质量。通过在两个方向上进行搜索并结合得分，
        #可以更好地捕捉到目标序列的特征和规律。