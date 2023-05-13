import torch
import numpy as np

def reqiregrad_set():
    xx = torch.tensor([1, 2, 4], dtype = torch.float, requires_grad=True)
    aa = np.array([1, 2, 3])
    yy = torch.from_numpy(aa).to(torch.float)
    yy.requires_grad = True
    print("tensor requires_grad: ", yy.requires_grad)

# 全连接层计算梯度 tiny
def autograd_demo():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True)    
    z = torch.matmul(x, w)+b # 全连接层
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    loss.backward() # 反向传播：求梯度
    print(w.grad)
    print(b.grad)
    print(y.grad) # 
    print(z.grad) # 中间结果不保存
    print("z requires_grad: ", z.requires_grad) #
    
def internal_grad_demo():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True)    
    z = torch.matmul(x, w)+b # 全连接层
    print("z shape: ", z.shape)
    output_grad = torch.randn_like(z)
    z.backward(output_grad)
    
def set_no_grad():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True) 
    z = torch.matmul(x, w)+b
    print("requires_grad: ", z.requires_grad)
    
    # torch.set_grad_enabled(False) # 全局设置 requires_grad = False
    
    with torch.no_grad():
        z = torch.matmul(x, w)+b
    print("requires_grad: ", z.requires_grad)
    
if __name__ == "__main__":
    # reqiregrad_set()
    # autograd_demo()
    # internal_grad_demo()
    set_no_grad()
    print("run autograd_demo.py successfully !!!")