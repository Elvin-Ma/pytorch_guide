import torch
import numpy as np

def tensor_create():
  # 方式1
  # data = [[1, 2],[3, 4]] # python list
  # x_data = torch.tensor(data)
  # x_data2 = torch.tensor((1, 2, 3))
  # # x_data3 = torch.tensor({"a": 5}) # fail
  # print("x_data2: ", x_data2)
  
  # 方式2
  # data = torch.ones(1, 2, 3)
  # data1 = torch.zeros(1, 3,4)
  # data2 = torch.randn(3, 4, 5)
  # data3 = torch.eye(4, 5)
  # data4 = torch.randint(5, (2, 10))
  # print("data type: ", type(data4))
  # print("data2: ", data4)
  
  # 方式3
  # data0 = torch.Tensor([1, 2, 3])
  # data1 = torch.ones_like(data0)
  # data2 = torch.empty_like(data0)
  # data3 = torch.empty_like(data0)
  # print("data: ", data2)
  
  # 方式4
  np_array = np.array([1, 2, 3])
  tensor_numpy = torch.from_numpy(np_array)
  # tensor_numpy2 = torch.Tensor(np_array) # deepcopy 了一份数据
  np_array[0] = 100
  # data_numpy = tensor_numpy.numpy()
  # print("data numpy: ", type(data_numpy))
  print("numpy tensor: ", tensor_numpy)

def tensor_struct():
  r'''
    meta_data / raw_data
  ''' 
  nd_array = np.array([[1, 2, 3], [4, 5, 6]])
  # tensor = torch.tensor(nd_array) # deep copy
  tensor = torch.from_numpy(nd_array)
  # meta_data
  # print("shape:", tensor.shape) #meta data
  # print("dtype: ", tensor.dtype) # met
  # print("stride: ", tensor.stride())
  # print("device: ", tensor.device)
  # .... 其它参考 /lib/python3.8/site-packages/torch/_C/__init__.pyi

  # raw data
  print("pytorch data: \n", tensor)
  # print("pytorch raw data: \n", tensor.storage())
  print("numpy raw data_ptr: ", nd_array.ctypes.data)
  print("pytroch raw data_ptr: ", tensor.data_ptr())
  
  print("numpy data id", id(nd_array))
  print("pytorch data id", id(tensor))
  
  tensor2 = tensor.reshape(1, 6)
  print("tensor id: ", id(tensor))
  print("tensor2 id: ", id(tensor2))
  print("tensor pointer addr: ", tensor.data_ptr())
  print("tensor2 pointer addr: ", tensor2.data_ptr())
  
def tensor_view():
  a = torch.arange(24).reshape(3, 8)
  b = a.T
  print("a shape: ", a.shape)
  print("b shape: ", b.shape)
  
  print("a stride: ", a.stride())
  print("b stride: ", b.stride())
  # print("a value: ", a)
  # print("b value: ", b)
  print("a storage:", a.data_ptr())
  print("b storage:", b.data_ptr())
  # print("a.shape: ", a.shape)
  # print("b.shape: ", b.shape)
  c = b[0][0]
  c = 100
  # c[1] = 100
  print("c value: ", a)
  
def stride_demo():
  a = torch.arange(24).reshape(3, 8)
  b = a.permute(1, 0).contiguous()
  c = b.reshape(4, 6)
  print("b shape: ", b.shape)
  print("c shape: ", c.shape)
  print("c data_ptr: ", c.data_ptr())
  print("b data_ptr: ", b.data_ptr())
  # print("c stride: ", c.stride())
  # print("b stride: ", b.stride())
  # d = a.reshape(4, 6)
  # print("c value: ", c.shape)
  # c = a.transpose(0, 1)
  # print("c shape: ", c.shape)
  # b = a.reshape(4, 3)
  # print("a shape: ", a.shape)
  # print("a stride: ", a.stride())
  # print("b shape: ", b.shape)
  # print("b stride: ", b.stride())
  # print("b shape: ", b.shape)
  # print("b stride: ", b.stride())
  
def numpy_with_torch_tensor():
  ndarray = np.array([1, 3, 4])
  tensor = torch.tensor(ndarray)
  tensor_from_numpy = torch.from_numpy(ndarray)
  
  print("numpy data_ptr: ", ndarray.ctypes.data)
  print("torch data_ptr: ", tensor.data_ptr())
  print("tensor_from numpy data_ptr: ", tensor_from_numpy.data_ptr())

  ndarray_from_torch = tensor_from_numpy.numpy()
  print("ndarray_from_torch data_ptr: ",   ndarray_from_torch.ctypes.data)

def tensor_to_demo():
  tensor = torch.ones(4, 5)
  print("tensor dtype: ", tensor.dtype)
  # print("tensor device: ", tensor.device)
  
  tensor_0 = tensor.to(torch.int32).to("cuda:0") # 数据的搬迁 h2d: h: host d:device(gpu)
  # print("tensor device: ", tensor_0.device)
  
  # tensor_1 = tensor.cuda(0)
  # print("tensor1 dtype: ", tensor_1.device)
  
  # if torch.cuda.is_available():
  #   device = torch.device("cuda:0")
  # else:
  #   device = torch.device("cpu")
    
  # tensor_2 = tensor.to(device)
  # tensor_3 = tensor.to(device)
  
  # #dtype 转化
  # a = np.array([1.0, 2, 3])
  # tt = torch.Tensor([1, 2, 3])
  # print("tt dtype: ", a.dtype)
  # tensor_4 = tensor.to(torch.float32)

  # # 1. 获取tensor0 的设备， 2 完成设备上的数据copy
  tensor_5 = tensor.to(tensor_0).cpu()
  tensor_6 = tensor.cuda() + tensor_5
  print(tensor_5.device)
  print(tensor_5.dtype)
  
def id_with_ptr():
  tensor_0 = torch.ones(4, 6)
  tensor_1 = tensor_0.reshape(3, 8)
  
  '''
  # print("tensor0 data_ptr: ", id(tensor_0[0, 0]))
  # print("tensor1 data_ptr: ", id(tensor_0[0][0]))
  
  # print("tensor0 data_ptr: ", id(tensor_0[0, 0]))
  # print("tensor1 data_ptr: ", id(tensor_0[0][0]))
  
  # print("tensor_0 data_ptr: ", tensor_0.data_ptr())
  '''
  
  # meta_data 发生改变，id 就会变
  # print("tensor_0 id: ", id(tensor_0))
  # print("tensor_1 id: ", id(tensor_1))
  # # ptr 只关注raw_data
  # print("tensor_0 data_ptr: ", tensor_0.data_ptr())
  # print("tensor_1 data_ptr: ", tensor_1.data_ptr())
  
  # print("tensor_0 data_ptr: ", tensor_0.data_ptr)
  # print("tensor_1 data_ptr: ", tensor_1.data_ptr)
  
def broadcast_demo():
  a = torch.rand(3, 1, 4, 2, 1)
  b = torch.randn(4, 1, 1, 5)
  c = a + b
  
def inplace_demo():
  a = torch.ones(3, 5)
  b = a.add_(5) # inplace 操作 一定要小心
  c = a.add(10)
  print("tensor_a data_ptr: ", a)
  print("tensor_b data_ptr: ", b)
  print("tensor_c data_ptr: ", c)
   
# def stride_demo():
#   a = torch.randn(2, 3)
#   b = a.reshape(3, 2)
#   c = a.T
#   d = c.contiguous() # 1. 重新申请了内存 2、数据重排
  
#   print("a storage: \n", a.storage())
#   print("c storage: \n", c.storage())
#   print("d storage: \n", d.storage())
  
#   print("a data_ptr: ", a.data_ptr())
#   print("c data_ptr: ", c.data_ptr())
#   print("d data_ptr: ", d.data_ptr())
  
#   # d = torch.rand(3, 4, 5)
#   # e = d.permute(0, 2, 1)
  
#   # print("a shape: ", a.shape)
#   # print("a stride: ", a.stride())
#   # print("b shape: ", b.shape)
#   # print("b stride: ", b.stride())
#   # print("c shape: ", c.shape)
#   # print("c stride: ", c.stride())
  
def reshape_vs_view():
  a = torch.randn(4, 6)
  b = a.T
  c = b.reshape(3, 8)
  d = b.contiguous().view(3, 8)
  
def tensor_api_demo():
  a = torch.Tensor()
  pass
    
if __name__ == "__main__":
  # tensor_create()
  # raw_data_demo()
  # tensor_struct()
  # tensor_view()
  # stride_demo()
  # numpy_with_torch_tensor()
  # tensor_to_demo()
  # id_with_ptr()
  # broadcast_demo()
  inplace_demo()
  # stride_demo()
  # reshape_vs_view()
  print("run tensor_demo.py successfully !!!")
  