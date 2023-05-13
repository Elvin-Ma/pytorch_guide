# pytorch Tensor guide

## pytorch 相关代码
- /site-packages/torch/_C/_VariableFunctions.pyi --> tensor 的c++接口
- /lib/python3.8/site-packages/torch/_tensor.py

## pytorch tensor 的 n 中创建方式
- python list / tuple --> 直接创建 Tensor
- numpy 方式创建
- torch 本身自带的一些创建方式

## tensor 的构成
- meta_data : 描述一个tensor(shape/dtype/ndim/stride)
- raw_data : 内存中的数据，以裸指针；
  *ndarray.ctypes.data numpy 获取方式
  *tensor.data_ptr() torch tensor的获取方式

## pytorch tensor 与 numpy的 相互转化 与 数据共享
- from_numpy 形成的 torch.tensor 和numpy 公用一套数据
- .numpy() 返回到 numpy 格式

## to 的理解
- 数据类型的转化 int --> float
**数据要从cpu copy 到 gpu上：h2d**
- 设备的转化 host(cpu) 和 device(GPU)的 概念
**d2h**
- tensor.cpu()

## Tensor 的 id 和 ptr
- id: tensor 的地址 (meta data 和 raw data 一个变就会变)
- data_ptr: 数据的地址(meta_data 变换不影响它)

## 理解tensor的视图

## tensor的计算方式：matmul 、 pointwise 、 elementwise

## broadcast

## inplace 操作

## 扩展
*_tensor.py*
*data到底存储在哪里呢？*
