# pytorch Tensor guide

## pytorch tensor 的 n 中创建方式

## pytorch tensor 与 numpy的 相互抓好

## 如何最正确的理解 Tensor
** raw data **
- numpy
  * ndarray.ctypes.data
  * data_view = memoryview(arr) && bytes(data_view)
  * raw_data = arr.tobytes() && int_data=np.frombuffer(raw_data, dtype=np.int64) *

- pytorch
  * tensor.data_ptr()

** meta data **
- shape
- dim
- dtype
- device
- stride

## Tensor 的 id 和ptr


## 理解tensor的视图

## tensor的计算方式：matmul 、 pointwise 、 elementwise

## broadcast

## inplace 操作

## 扩展
*_tensor.py*
*data到底存储在哪里呢？*
