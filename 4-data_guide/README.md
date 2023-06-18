# 数据准备

# 两个重要的类（pytorch接口）
- dataset(/site-packages/torch/utils/data/dataset.py)
```python
from torch.utils.data import Dataset
```
- dataloader(site-packages/torch/utils/data/dataloader.py)
```python
from torch.utils.data import DataLoader
```

# torchvision 是什么
- 它和 pytorch 是并列关系，它不属于pytorch的一部分；
- 但是 torchvision 是依赖于pytorch的；
- 计算机数据相关方面的工具集；
- torchtext、torchaudio 都类似；
- VisionDataset(data.Dataset)

# pytorch 官方 dataset:
```python
class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])
```
- /site-packages/torch/utils/data/dataset.py
- 继承官方的dataset
- 实现自己的 __getitem__ 和 __len__ 方法
- 数据集和label 存放到__init__里（self.data self.label）
- __len__ 返回数据集总的长度
- __getitem__: 以 batch = 1 
1. 从self.data 里拿到 具体的某一个index 的数据；
2. 从self.label 中拿到 对应index 的label；
3. 数据转换：转化为我们需要的数据形式（数据增强的过程就在这里）；
4. 把 data 和 label 返回

# dataloader
- 多batch：一次调用多个__getitem__ 来实现多batch的添加；
- 完成训练时候 每次batch的数据加载工作；
- 用户需要输入 dataset，以及其它参数的一些设置

# 应用官方工具
## 三大方向
[pytorch 主仓库](https://github.com/pytorch)
- vision
- text
- audio

## torchvison
- datasets
- models(预训练的model)
- tranform






