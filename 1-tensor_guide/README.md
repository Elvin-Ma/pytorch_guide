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
- 视图要拿最终的raw data来看， id 经常出现问题。


## tensor的计算方式：matmul 、 pointwise 、 elementwise
- 矩阵形式的计算：matmul、rashpe/ permute/ transpose
- pointewise / elementwise (元素之间的操作)
- broadcast: 1. shape 右对齐；2. 对应dim 有一个是1 或者 相同；

## inplace 操作
- 原地操作（raw data 不变），新计算出来的结果，替换掉原来结果；
- 节约内存，提升性能
- 不推荐使用，除非非常了解。

## 理解tensor的视图
**reshape、permute、transpose、view**
- reshape 和 view 是一组：；
- permute 和 transpose 是一组：数据不连续的现象；
**这四个算子raw data 都一样**
**通常我们根据shape 就可以推断出 stride**
**但是 transpose 和 permute 不能通过shape 来推断出来**
**此时发生数据不连续**
- 影响我们计算结果吗？no
- 性能不好；
- 我们应该怎么办？contiguous

**contiguous发生了什么**
*1. 重新申请了内存*
*2.数据重排了*

**reshape vs view**
- reshape 不管数据是否连续，都可以用；
- view 智能用于数据连续的情况下；
- 如果必须用view：数据要先进行 contiguous
- 数据不连续的情况下：reshape = contiguous + view
- reshape ：数据不连续去情况下 reshape 自动进行 数据copy 和 reorder
- contiguous：本身也就是 数据copy 和重排

## 扩展
*_tensor.py*
*data到底存储在哪里呢？*
