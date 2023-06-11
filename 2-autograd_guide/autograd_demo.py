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
    torch.manual_seed(0)
    global y 
    x = torch.ones(5, requires_grad=True)
    y = torch.randn(5, 5, requires_grad=True)
    b = torch.randn_like(x)
    
    grad_list = []
    
    def aa(grad):
        grad_list.append(grad)
    
    # print("==========1: ", y)
    for i in range(10000):
        # global y
        # y.requires_grad=True
        # y.register_hook(aa)
        y.retain_grad()
        z = torch.matmul(y, x) + b # linear layer    
        output = torch.sigmoid(z)
        label = torch.Tensor([0, 0, 1, 0, 0])
        loss = (output-label).var() # l2 loss
        loss.backward()
        # a = y.grad
        y = y - 0.1 * y.grad
        # a = y.grad
        # b = 0.1 * a
        # if i < 90:
        #    y = y - 0.1*grad_list[-1]
        # elif i >= 90:
        # print("==============: ", i)
        # print("pre: \n", y)
        # y1 = y.detach()
        # y = y1 - 0.2*grad_list[-1]
        # a = y.grad
        # b = y.detach()
        
        # y.requires_grad = True
        # y.requires_grad_(True)
        # print("beh: \n", y)
        # print(y)
        # y.retain_grads = True
        # y.requires_grad = True
        
        print("loss: ", loss)
        # torch.Tensor
        # print("x grad: ", x.grad)
    # print("==========2: ", y)
    
    # x = torch.ones(5)  # input tensor
    # y = torch.zeros(3)  # expected output
    # w = torch.randn(5, 3, requires_grad=True) # requires_grad
    # b = torch.randn(3, requires_grad=True)    
    # z = torch.matmul(x, w)+b # 全连接层
    
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    # print(f"Gradient function for z = {z.grad_fn}")
    # print(f"Gradient function for loss = {loss.grad_fn}")
    # loss.backward() # 反向传播：求梯度
    # print(w.grad)
    # print(b.grad)
    # print(y.grad) # 
    # print(z.grad) # 中间结果不保存
    # print("z requires_grad: ", z.requires_grad) #
    
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
    x = torch.ones(5, requires_grad=True)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True) 
    z = torch.matmul(x, w)+b
    # print("requires_grad: ", z.requires_grad)
    z.backward(torch.randn_like(z))
    print("x grad: ", x.grad)
    
    # torch.set_grad_enabled(False) # 全局设置 requires_grad = False
    
    with torch.no_grad():
        z = torch.matmul(x, w)+b
    print("requires_grad: ", z.requires_grad)
    
def grad_sum():
    # torch.seed()
    x = torch.ones(5)  # input tensor
    label = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True)    
    output = torch.matmul(x, w)+b # 全连接层 
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label)
    loss.backward(retain_graph=True) # 反向传播：求梯度
    print(f"Grad for w first time = {w.grad}")
    print(f"Gradient function for z = {output.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    w.grad.zero_()
    loss.backward(retain_graph=True)
    print(f"Grad for w second time = {w.grad}")
    
    
def hook_demo():
    v = torch.tensor([0., 0., 0.], requires_grad=True)
    h = v.register_hook(lambda grad: grad * 1 + 2)  # double the gradient
    v.backward(torch.tensor([1., 2., 3.]))
    print("v grad: ", v.grad)
    
def get_inter_grad():
    z_grad = []
    # def get_grad(grad):
    #     z_grad.append(grad)
        
    torch.manual_seed(0)
    x = torch.ones(5)  # input tensor
    label = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True)    
    output = torch.matmul(x, w)+b # 全连接层
    output.retain_grad()
    # output.register_hook(get_grad)    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label)
    loss.backward(retain_graph=True) # 反向传播：求梯度
    print("output grad: ", output.grad)   
    
class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print("==============")
        result, = ctx.saved_tensors
        return grad_output * result
    
def custom_demo():
    input = torch.randn(5, 6)
    input.requires_grad=True
    output = Exp.apply(input)
    output.backward(torch.rand(5, 6))
    print("output: ", input)
    print("output grad: ", input.grad)
    
def outer(f):
    def inner(*args, **kargs):
        inner.co += 1
        print("=======: ", inner.co)
        return f(*args, **kargs)
    print("========== aaa")
    inner.co = 0
    return inner

@outer
def f():
    pass
    
if __name__ == "__main__":
    # reqiregrad_set()
    autograd_demo()
    # internal_grad_demo()
    # set_no_grad()
    # grad_sum()
    # hook_demo()
    # get_inter_grad()
    # custom_demo()
    print("run autograd_demo.py successfully !!!")