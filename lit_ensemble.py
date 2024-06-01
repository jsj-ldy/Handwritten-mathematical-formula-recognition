import zipfile
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from bttr.datamodule import Batch, vocab, vocab_size
from bttr.lit_bttr import LitBTTR
from bttr.utils import ExpRateRecorder, Hypothesis, to_tgt_output
#这个类的作用是将多个预训练的模型组合成一个集成模型，通过集成多个模型的预测结果来提高性能，
# 例如在推理阶段通过对多个模型的预测结果进行平均或投票来产生最终的预测。
#用于构建一个模型集成器，该集成器由多个预训练的模型组成
class LitEnsemble(pl.LightningModule):#接受一个参数 paths，该参数是包含预训练模型文件路径的列表。
    def __init__(self, paths: List[str]):
        super(LitEnsemble, self).__init__()

        self.models = nn.ModuleList()
        for p in paths:
            #通过遍历 paths 中的路径，加载每个预训练模型，并将其添加到 self.models 中
            model = LitBTTR.load_from_checkpoint(checkpoint_path=p)
            model = model.eval()
            self.models.append(model)

        self.beam_size = self.models[0].hparams.beam_size
        self.max_len = self.models[0].hparams.max_len
        self.alpha = self.models[0].hparams.alpha
        self.recorder = ExpRateRecorder()

    def test_step(self, batch: Batch, _):
        hypotheses = self.beam_search(batch.imgs, batch.mask)#假设序列
        #最佳假设序列
        best_hypo = max(hypotheses, key=lambda h: h.score / (len(h) ** self.alpha))
        #获取真实标签序列
        indices = batch.indices[0]
        #预测标签序列
        indices_hat = best_hypo.seq
        #更新性能指标记录器，也就是正确率，ExpRateRecorder 通过比较预测序列和真实序列之间的编辑距离来计算正确率。
        self.recorder(indices_hat, indices)
        #文件名和预测标签
        return {
            "fname": batch.img_bases[0],
            "pred": vocab.indices2label(indices_hat),
        }
    #你正在处理测试集的输出并计算最终的表现度量（如成功率），将测试得出结果保存在文件中
    def test_epoch_end(self, outputs) -> None:
        exp_rate = self.recorder.compute()

        print(f"ExpRate: {exp_rate}")
        print(f"length of total file: {len(outputs)}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for d in outputs:
                content = f"%{d['fname']}\n${d['pred']}$".encode()
                with zip_f.open(f"{d['fname']}.txt", "w") as f:
                    f.write(content)

    def beam_search(self, img, mask):
        #源掩码列表 src_mask_list，包含了每个模型对应的编码器输出的掩码。
        src_mask_list = [m.bttr.encoder(img, mask) for m in self.models]
        #进行束搜索，得到 l2r 方向的候选序列
        l2r_hyps = self.ensemble_beam_search(
            src_mask_list=src_mask_list,
            direction="l2r",
            beam_size=self.beam_size,
            max_len=self.max_len,
        )
        #候选序列之间的交叉正确率得分，将其作为指导信息来综合评估每个候选序列的质量
        self.ensemble_cross_rate_score(
            src_mask_list=src_mask_list,
            hypotheses=l2r_hyps,
            direction="r2l",
        )
        #进行束搜索，得到 r2l 方向的候选序列
        r2l_hyps = self.ensemble_beam_search(
            src_mask_list=src_mask_list,
            direction="r2l",
            beam_size=self.beam_size,
            max_len=self.max_len,
        )
        #候选序列之间的交叉正确率得分，将其作为指导信息来综合评估每个候选序列的质量
        self.ensemble_cross_rate_score(
            src_mask_list=src_mask_list,
            hypotheses=r2l_hyps,
            direction="l2r",
        )
        return l2r_hyps + r2l_hyps
#区别在于把每个预测模型的候选序列都的出来了
    def ensemble_beam_search(
        self,
        src_mask_list: List[Tuple[torch.Tensor, torch.Tensor]],#源序列掩码
        direction: str,#方向
        beam_size: int,#束大小
        max_len: int,
    ) -> List[Hypothesis]:
        """search result for single image with beam strategy

        Args:
            src_mask_list: [([1, len, d_model], [1, len])]
            direction (str):
            beam_size (int): beam size
            max_len (int): max length for decode result

        Returns:
            List[Hypothesis(seq: [max_len])]: list of hypotheses(no order)
        """
        #确定方向
        assert direction in {"l2r", "r2l"}

        if direction == "l2r":
            start_w = vocab.SOS_IDX
            stop_w = vocab.EOS_IDX
        else:
            start_w = vocab.EOS_IDX
            stop_w = vocab.SOS_IDX
        #初始化候选序列
        hypotheses = torch.full(
            (1, max_len + 1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=self.device,
        )
        hypotheses[:, 0] = start_w
        #初始化得分序列
        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)
        completed_hypotheses: List[Hypothesis] = []

        t = 0
        # 在循环中，模型根据当前的假设，源序列，掩码生成下一时刻的预测输出，并计算对应的得分
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)#当前的假设数量
            #假设数量<=束的大小
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"
            #初始化
            prob_sum = torch.zeros(
                (hyp_num, vocab_size),#指定了张量的形状，hyp_num表示假设数量，vocab_size表示词汇表的大小
                #hyp_num 行和 vocab_size 列
                dtype=torch.float,
                device=self.device,
            )
            for i, m in enumerate(self.models):#对模型列表中的每个模型执行一系列操作
                src, src_mask = src_mask_list[i]#获取当前与索引i对应的源序列，和源序列掩码
                exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)#重复hyp_num次，并扩展他们
                exp_src_mask = repeat(src_mask.squeeze(0), "s -> b s", b=hyp_num)#重复hyp_num次，并扩展他们
                #通过解码器，获取特定时间步t的输出
                decode_outputs = m.bttr.decoder(exp_src, exp_src_mask, hypotheses)[
                    :, t, :
                ]
                #在所有模型中累积概率
                prob_sum = prob_sum + torch.softmax(decode_outputs, dim=-1)
                #对输出的张量沿最后一个维度应用softmax函数，获取概率
            #计算的模型输出的累积概率之和，然后除以模型数量来获取平均概率，取对数得到对数概率
            log_p_t = torch.log(prob_sum / len(self.models))
            #还需要生成的假设数量
            live_hyp_num = beam_size - len(completed_hypotheses)
            #将得分扩展，获得与词汇表匹配的大小
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e=vocab_size)
            #将假设得分和对数概率相加，以便能够与词汇表大小匹配
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            #找到最高的live_hyp_num个得分及其对应的位置。
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                continuous_hyp_scores, k=live_hyp_num
            )
            #从位置中分离出先前的假设索引和新添加的单词索引。
            prev_hyp_ids = top_cand_hyp_pos // vocab_size
            hyp_word_ids = top_cand_hyp_pos % vocab_size
            #增加时间不
            t += 1
            #初始化新生成的假设
            new_hypotheses = []
            #初始化得分列表
            new_hyp_scores = []
            #根据先前的假设和新预测的单词来更新假设列表
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                #cand_new_hyp_score张量的数值表示，同时确保不再计算梯度。
                # 将候选假设得分从计算图中分离出来并转换为Python数值。
                #detach()方法用于创建一个新的张量，其内容与原始张量相同，但不再具有与计算图的连接，即它不再需要梯度计算。
                #用于将包含单个元素的张量转换为Python数值（标量），并返回这个数值。
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                #将新预测的单词索引添加到相应的先前假设中，形成一个新的假设序列。
                hypotheses[prev_hyp_id, t] = hyp_word_id
                #完成一个假设
                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(
                            #选择了对应先前假设的序列部分，但忽略了序列的起始标记 START_W。
                            #表示从序列的第二个位置（排除起始标记）到第 t-1 个位置
                            seq_tensor=hypotheses[prev_hyp_id, 1:t]
                            .detach()#.detach() 方法将该切片张量从计算图中分离出来，确保不再具有梯度信息
                            .clone(),#创建副本
                            # remove START_W at first
                            #这样做是为了避免在后续计算中修改原始张量。
                            score=cand_new_hyp_score,
                            direction=direction,
                        )
                    )
                else:
                    #hyp_word_id 不等于停止词 stop_w，则表示新假设仍在进行中
                    #将先前假设的克隆添加到 new_hypotheses 列表中
                    new_hypotheses.append(hypotheses[prev_hyp_id].detach().clone())
                    #并将对应的候选假设得分 cand_new_hyp_score 添加到 new_hyp_scores 列表中
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break
            #将 new_hypotheses 列表中的张量沿着指定的维度（dim=0，表示沿着第一个维度）进行堆叠
            #这个张量包含了所有新的假设。
            hypotheses = torch.stack(new_hypotheses, dim=0)
            #将new_hyp_scores列表转换为PyTorch张量
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )
        #如果在一次迭代中没有生成任何新的假设，表示是第一次生成假设，
        #则将当前最佳假设作为已完成的假设
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=hypotheses[0, 1:].detach().clone(),#当前最佳假设的序列部分
                    score=hyp_scores[0].detach().item(),#当前最佳假设的得分
                    direction=direction,
                )
            )

        return completed_hypotheses

    def ensemble_cross_rate_score(
        self,
        #第一个张量似乎表示形状为 [1, len, d_model] 的源序列，
        #第二个张量似乎表示形状为 [1, len] 的掩码。
        src_mask_list: List[Tuple[torch.Tensor, torch.Tensor]],
        #某个模型生成的假设
        hypotheses: List[Hypothesis],
        #模型方向
        direction: str,
    ) -> None:
        """give hypotheses to another model, add score to hypotheses inplace
        Args:
            src_mask_list: [([1, len, d_model], [1, len])]
            hypotheses (List[Hypothesis]):
            direction (str): one of {"l2r", "r2l"}
        """
        #每个假设序列，从hypotheses 列表中提取索引
        indices = [h.seq for h in hypotheses]
        #将这些索引转换为目标序列和输出
        tgt, output = to_tgt_output(indices, direction, self.device)
        #初始化
        b, length = tgt.size()
        prob_sum = torch.zeros(
            (b, length, vocab_size), dtype=torch.float, device=self.device
        )
        for i, m in enumerate(self.models):
            #获取当前与索引i对应的源序列，和源序列掩码
            src, src_mask = src_mask_list[i]
            #不断进行复制，适配当前的假设数量，准备数据
            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=b)
            exp_src_mask = repeat(src_mask.squeeze(0), "s -> b s", b=b)
            #模型预测的目标序列。
            output_hat = m.bttr.decoder(exp_src, exp_src_mask, tgt)
            #在解码过程中累积每个时间步的概率分布
            prob_sum = prob_sum + torch.softmax(output_hat, dim=-1)
        #为了计算平均概率的对数值，用于后续损失值的计算
        log_p = torch.log(prob_sum / len(self.models))

        flat_hat = rearrange(log_p, "b l e -> (b l) e")
        flat = rearrange(output, "b l -> (b l)")
        # PyTorch 中的负对数似然损失函数（nll_loss）来计算损失值。
        # ignore_index 参数指定了需要忽略的索引，通常用于填充标记。
        #reduction="none" 参数表示不对损失进行归约，即每个样本都有一个对应的损失值。
        loss = F.nll_loss(flat_hat, flat, ignore_index=vocab.PAD_IDX, reduction="none")
        #这一步是对损失沿着最后一个维度（即序列长度维度）进行求和，得到每个样本的总损失。
        loss = rearrange(loss, "(b l) -> b l", b=b)
        loss = torch.sum(loss, dim=-1)
        #
        for i, length in enumerate(loss):
            #因为通常情况下，得分越高表示模型效果越好，
            #但损失值越低表示模型效果越好，
            # 所以取负值是为了将损失值转换为得分，使得得分越高表示模型效果越好。
            score = -length
            #将每个样本的损失作为得分累积到假设对象中，以便后续对假设进行排序或其他操作。
            hypotheses[i].score += score
