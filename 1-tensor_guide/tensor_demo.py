import torch
import numpy as np

def tensor_create():
  data = [[1, 2],[3, 4]]
  x_data = torch.tensor(data)
  
  np_array = np.array(data)
  x_np = torch.from_numpy(np_array)

  x_ones = torch.ones_like(x_data) # retains the properties of x_data
  print(f"Ones Tensor: \n {x_ones} \n")

  x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
  print(f"Random Tensor: \n {x_rand} \n")
  
  shape = (2,3,)
  rand_tensor = torch.rand(shape)
  ones_tensor = torch.ones(shape)
  zeros_tensor = torch.zeros(shape)

  print(f"Random Tensor: \n {rand_tensor} \n")
  print(f"Ones Tensor: \n {ones_tensor} \n")
  print(f"Zeros Tensor: \n {zeros_tensor}")

def tensor_struct():
  r'''
    meta_data / raw_data
  ''' 
  tensor = torch.tensor([1, 2, 3])
  # meta_data
  print("shape:", tensor.shape)
  print("dtype: ", tensor.dtype)
  print("stride: ", tensor.stride())
  # .... 其它参考 /lib/python3.8/site-packages/torch/_C/__init__.pyi

  # raw data
  print("pytorch raw data: ", tensor.storage())
  print("pytroch raw data_ptr: ", tensor.data_ptr())

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
  print("tensor dtype: ", tensor.device)
  
  tensor_0 = tensor.to("cuda:0")
  print("tensor dtype: ", tensor_0.device)
  
  tensor_1 = tensor.cuda()
  print("tensor1 dtype: ", tensor_1.device)
  
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
  else:
    device = torch.device("cpu")
    
  tensor_2 = tensor.to(device)
  tensor_3 = tensor.to(device)
  
  #dtype 转化
  tensor_4 = tensor.to(torch.float32)

  # 1. 获取tensor0 的设备， 2 完成设备上的数据copy
  tensor_5 = tensor.to(tensor_0)
  
def id_with_ptr():
  tensor_0 = torch.ones(4, 6)
  tensor_1 = tensor_0.reshape(3, 8)

  # meta_data 发生改变，id 就会变
  print("tensor_0 id: ", id(tensor_0))
  print("tensor_1 id: ", id(tensor_1))
  # ptr 只关注raw_data
  print("tensor_0 data_ptr: ", tensor_0.data_ptr())
  print("tensor_1 data_ptr: ", tensor_1.data_ptr())
  
  print("tensor_0 data_ptr: ", tensor_0.data_ptr)
  print("tensor_1 data_ptr: ", tensor_1.data_ptr)
  
if __name__ == "__main__":
  # raw_data_demo()
  # tensor_struct()
  # numpy_with_torch_tensor()
  # tensor_to_demo()
  id_with_ptr()
  print("run tensor_demo.py successfully !!!")