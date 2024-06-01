import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from bttr.datamodule.vocab import CROHMEVocab

vocab = CROHMEVocab()
#data是一个列表，每个元素是三元组，文件名，图像数据（PIL库的Image对象），一个字符串列表
Data = List[Tuple[str, Image.Image, List[str]]]
#根据gpu确定
MAX_SIZE = 32e3  # change here accroading to your GPU memory

# load data
#将输入的数据按照指定的规则进行批次化，也就是把整体的数据分为一个小批次一个小批次。
def data_iterator(
    data: Data,
    batch_size: int,#每个批次中样本的数量
    batch_Imagesize: int = MAX_SIZE,#一个批次中图像的累计最大尺寸，这个限制可以确保每个批次的总体大小不会超过预设的阈值，有助于控制内存或显存
    maxlen: int = 200,#标签的最大长度限制
    maxImagesize: int = MAX_SIZE,#单个图像的大小
):
    #初始化变量
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0
    #按照图像的大小对data列表的元组进行升序排序，图像特征的宽度乘以图像特征的高度，按整个图像特征的面积大小定的
    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])
    #按从小到大排序
    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = transforms.ToTensor()(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)#当前批次的图像大小的总和
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[1]} x {fea.shape[2]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            #根据当前累计的图像大小判断是否形成一个完整的批次，
            # 如果是则将当前批次的文件名，特征和标签添加到当前的批次中，
            # 并更新累计的图像大小
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))
#函数返回一个列表，每个元素是一个元组，包含一个批次中的文件名，特征和标签，就是一个图像对应一个元组，每一个元组里面包含
#文件名，特征和标签
#从文件中读取数据集dir-name表示压缩文件中要提取数据的目录名称
#从压缩文件中提取数据，用于创建数据集，其中每个元组包含图像名称，图像对象和相应的公式
def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()#遍历caption中每一行，对其进行解析，提取图像名称和相应的公式
        img_name = tmp[0]#图像名称
        formula = tmp[1:]#公式
        with archive.open(f"{dir_name}/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()#创建图像的副本，确保图像对象在内存中被完全加载，而不是惰性加载
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")
#返回元组，包括图像名称，图像和公式
    return data


@dataclass#这是一个叫做装饰器，
class Batch:#封装了图像数据相应的掩码以及其他相关信息，图像基础和索引
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:#可能对应批次大小
        return len(self.img_bases)

    def to(self, device) -> "Batch":#将imgs和mask张量移动到指定的设备上
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )

#collate_fn 函数的作用是将一个批次的样本数据整理成模型可以处理的格式，
# 并返回一个 Batch 对象。
def collate_fn(batch):
    assert len(batch) == 1#断言输入的batch的长度为1
    batch = batch[0]
    fnames = batch[0]#批次的第一个数据，名字
    images_x = batch[1]#批次的第二个数据，图像数据
    seqs_y = [vocab.words2indices(x) for x in batch[2]]
    #遍历批次数据的第三个元素，并使用方法将其转换为索引序列,也就是根据对应的单词得到其对应的索引
    heights_x = [s.size(1) for s in images_x]#图像的高度列表
    widths_x = [s.size(2) for s in images_x]#图像的宽度列表

    n_samples = len(heights_x)#有多少个例子
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)#初始化为全0张量
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)#初始化为一个全一的布尔型张量，用于表示图像数据的掩码
    for idx, s_x in enumerate(images_x):#迭代列表中的数据，同时获得索引和值
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x#在x张量的第idx的位置，将高度和宽度都限制在最大的值的范围内
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0#对应位置赋值为0表示有效数据

    #return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)
#处理后的文件名，图像数据，图像掩码和序列索引数据
#接受一个存档文件，一个文件夹路径和一个批次大小，从存档文件中提取数据，并将其组织成合适训练模型的形式，然后返回一个数据迭代器，该迭代器可以按批次迭代训练数据
def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)#返回元组，包括图像名称，图像和公式
    return data_iterator(data, batch_size)
#函数返回一个列表，每个元素是一个元组，包含一个批次中的文件名，特征和标签，就是一个图像对应一个元组，每一个元组里面包含
#准备和加载数据
class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        # os.path.realpath(__file__) 返回当前脚本文件的绝对路径，不包括文件名。
        # os.path.dirname() 返回指定路径的目录名称部分。
        test_year: str = "2016",
        train:str="2014",
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()#调用父类的初始化方法
        assert isinstance(test_year, str)#检查test_year是不是字符串类型的
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Load data from: {self.zipfile_path}")
#根据指定的阶段构建不同用途的数据集并将他们存储在类的属性中

    def setup(self, stage: Optional[str] = None) -> None:#stage用于指定数据集的阶段，fit训练，test测试，None是默认
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = build_dataset(archive, self.train, self.batch_size)
                self.val_dataset = build_dataset(archive, self.test_year, 1)#验证数据集
            if stage == "test" or stage is None:
                self.test_dataset = build_dataset(archive, self.test_year, 1)#测试数据集
#训练的数据加载器
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,#表示在每个epoch开始时打乱数据集，增加模型训练的随机性
            num_workers=self.num_workers,#数据加载的子进程数量，数据加载的并行度
            collate_fn=collate_fn,#处理数据批次的函数

        )
#DataLoader 是 PyTorch 中用于封装数据加载逻辑的类，
# 它可以将数据集分成批次并进行加载，并且支持多线程加载和数据批处理。
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,

        )


if __name__ == "__main__":#检查当前模块是否是诸臣修改v运行
    from argparse import ArgumentParser

    batch_size = 2
    #用于解析命令行参数并解析用户在命令行中输入的参数。
    parser = ArgumentParser()
    #添加命令行参数
    parser = CROHMEDatamodule.add_argparse_args(parser)#添加用于解析命令行参数的参数选项
    #解析命令行参数
    args = parser.parse_args(["--batch_size", f"{batch_size}"])#解析命令行参数，手动指定batch_size的值
    #根据命令行参数来初始化类的实例
    dm = CROHMEDatamodule(**vars(args))#根据解析得到的参数创建一个实例
    dm.setup()#不同阶段的数据模块

    train_loader = dm.train_dataloader()#创建训练数据加载器
    for img, mask, tgt, output in train_loader:#只获得了第一个批次的数据
        break
