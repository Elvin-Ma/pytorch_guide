import torch
import numpy as np

def raw_data_demo():
  """
  bbbbbbbbbbbbbbbbbbbbb
  """
  ndarray = np.array([[2, 3, 4], [4, 5, 6]])
  tensor = torch.from_numpy(ndarray)
  ndarray[0][0] = 100
  print("=============ndarray ptr: ", ndarray.ctypes.data)
  print("==============storage: ", tensor.storage())
  print("==============data_ptr: ", tensor.data_ptr())


if __name__ == "__main__":
  raw_data_demo()
  print("run tensor_demo.py successfully !!!")