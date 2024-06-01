import os
from functools import lru_cache
from typing import Dict, List

#如果函数被多次调用，传入的参数一样则，直接用缓存中的数据，不再运行一次
@lru_cache()#使用最近最少使用缓存策略来缓存函数的结果
def default_dict():#
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")
#os.path.abspath(__file__) 返回当前脚本文件的绝对路径，包括文件名。
#os.path.dirname() 返回指定路径的目录名称部分。
#os.path.join() 将多个路径组合成一个路径。
#这个类主要用于文本处理任务中构建词汇表，并提供了单词到索引、索引到单词的转换功能
class CROHMEVocab:

    PAD_IDX = 0#填充
    SOS_IDX = 1#起始
    EOS_IDX = 2#结束

    def __init__(self, dict_path: str = default_dict()) -> None:
        self.word2idx = dict()#初始化一个空字典，存储单词到索引的映射关系
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX

        with open(dict_path, "r") as f:
            for line in f.readlines():
                w = line.strip()#方法用于去除字符串两端的空白字符
                self.word2idx[w] = len(self.word2idx)#每个单词对应一个索引
        #索引到单词的反向映射
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        # print(f"Init vocab with size: {len(self.word2idx)}")
#接受单词列表作为参数返回对应的索引列表
    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]
#通过索引列表作为参数返回对应的单词列表
    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]
#返回单词序列，单词之间用空格分隔
    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)
#返回词汇表的大小啊
    def __len__(self):
        return len(self.word2idx)
