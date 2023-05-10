# pytorch Tensor guide

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

# 扩展
*data到底存储在哪里呢？*
