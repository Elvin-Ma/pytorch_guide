# pytorch Tensor guide

## tensor 是什么
- 张量 （weight activationg）
- 多维数据，numpy --> ndarray
- pytorch 里最大的一个类（class）
- 属性、和方法

## 初始化一个tensor
- torch.Tensor(***) 调用 Tensor 这个类的init方法完成初始化
- torch.ones() 调用torch本身自带的一些函数来生成特殊类型的tensor
- torch.ones_like() # 也是调用torch 自带的函数来生成，shape等信息采用另外一种tensor的；
- numpy 与 tensor之间的互相转换：优先选用 from_numpy: 数据共享

# tensor 的结构
1. meta_data ：dtype、shape、dims、device、stride()
2. raw_data : 内存中的数据
3. data_ptr 查询的是我们raw_data;
4. id : raw_data 或者 mete_data 有一个改变就会改变

# tensor 的视图
- transpose storage 之后没有发生任何的变化，raw_data 没有变；
- stride: 正常情况下某一维度的stride的值，后几个维度相乘；

# reshape/view/permute/transpose
- 这四个算子(方法) raw_data 都没变；
- 变化的都是 shape 和 stride；
- reshape 和 view 可以改变维度的；
- permute/transpose 换轴之后就不满足这种规律（stride）了 --> 数据不连续 uncontiguous；
- reshape 和 view ： 在大部分情况下都是一样的；数据不连续的情况下就有区别了。
- view: 永远不会生成新的数据，永远是一个view，视图；
- reshape：如何可能返回一个视图的话，它就返回一个视图，如果出现数据不连续的情况导致返回不了视图，
- reshape就会返回一个新的tensor(新的一份raw_data)
- uncontiguous: 对我们硬件不有好的，会导致我们的性能下降；

## pytorch 相关代码
- /site-packages/torch/_C/_VariableFunctions.pyi --> tensor 的c++接口
- /lib/python3.8/site-packages/torch/_tensor.py

## 如何学习API（tensor api 举例）
- 软件工程中 api， file name 都不是随便取的；
- tensor的接口在哪里 --> _tensor.py 中；
- _tensor.py 中的 class Tensor  只是子类，
- 父类： class _TensorBase 在__init__.pyi 中
- 常用的属性：
**requires_grad、shape、dytpe、layout、ndim、grad_fn**
- 常用的方法
**pointewise类型的方法：abs、acos、add、addcdiv**
**投票函数：all any**
**bit相关的操作**
**clone**
**统计学上的运算：mean var median min max**
**backward：反向传播的时候使用**
**register_hook: 钩子函数**
** retain_grad: 保留梯度** 
** resize：通过插值的方式进行尺度缩放 **
** grad 查询梯度信息**

## tensor 的构成
- meta_data : 描述一个tensor(shape/dtype/ndim/stride)
- raw_data : 内存中的数据，以裸指针；
  *ndarray.ctypes.data numpy 获取方式
  *tensor.data_ptr() torch tensor的获取方式

## pytorch tensor 与 numpy的 相互转化 与 数据共享
- from_numpy 形成的 torch.tensor 和numpy 公用一套数据
- .numpy() 返回到 numpy 格式

## to 的理解
- 数据类型的转化: int --> float
**数据要从cpu copy 到 gpu上：h2d**
**数据从 gpu copy 到 cpu：d2h**
- 设备的转化 host(cpu) 和 device(GPU)的 概念
**d2h**
- tensor.cpu()
- 不同设备上的tensor，不能进行运算；

##自动选择后端
```python
if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")
```

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
