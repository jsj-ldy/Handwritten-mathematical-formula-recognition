from bttr.datamodule.datamodule import Batch, CROHMEDatamodule, vocab

vocab_size = len(vocab)#词汇表的大小

__all__ = [
    "CROHMEDatamodule",
    "vocab",
    "Batch",
    "vocab_size",
]
#这段代码的作用是将 Batch、CROHMEDatamodule、vocab 和 vocab_size
# 这几个对象或变量导入到模块中，并且定义了模块的公开接口，
# 以便其他模块可以使用 from module import * 语法导入这些接口。






