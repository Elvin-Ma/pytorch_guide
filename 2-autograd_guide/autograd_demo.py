import torch
import numpy as np

def reqiregrad_set():
    xx = torch.tensor([1, 2, 4], dtype = torch.float, requires_grad=True)
    aa = np.array([1, 2, 3])
    yy = torch.from_numpy(aa).to(torch.float)
    yy.requires_grad = True
    print("tensor requires_grad: ", yy.requires_grad)

# 全连接层计算梯度 tiny
def autograd_demo_v1():
    torch.manual_seed(0)
    x = torch.ones(5, requires_grad=True) # input
    w = torch.randn(5, 5, requires_grad=True) # weight
    b = torch.randn_like(x)
    
    grad_list = []
    
    def hook(grad):
        grad_list.append(grad)

    for i in range(100):
        # w.requires_grad=True # True 
        w.register_hook(hook)

        z = torch.matmul(w, x) + b # linear layer    
        output = torch.sigmoid(z)
        label = torch.Tensor([0, 0, 1, 0, 0])
        loss = (output-label).var() # l2 loss
        loss.backward()
        
        w = w - 0.2 * grad_list[-1] #why wrong? --> 我们不能改变非叶子节点的requires_grad
        # w1 = w.detach() # detach 
        # w = w1 - 0.2*grad_list[-1] # w 新建的tensor    
        print("loss: ", loss)

def autograd_demo_v2():
    torch.manual_seed(0)
    x = torch.ones(5, requires_grad=True)
    y = torch.randn(5, 5, requires_grad=True) # 叶子节点
    b = torch.randn_like(x)

    for i in range(100):
        # y.grad.zero_()
        y.retain_grad()
        # print("===========y.grad: ", y.grad)
        z = torch.matmul(y, x) + b # linear layer    
        output = torch.sigmoid(z)
        label = torch.Tensor([0, 0, 1, 0, 0])
        loss = (output-label).var() # l2 loss
        loss.backward()
        y = y - 0.2 * y.grad # y 新的y --> 已经完成了梯度清0；
        print("loss: ", loss)
        
# 全连接层计算梯度 tiny
def autograd_demo_v3():
    torch.manual_seed(0)
    x = torch.ones(5, requires_grad=True)
    y = torch.randn(5, 5, requires_grad=True) # 叶子节点
    b = torch.randn_like(x)

    for i in range(100):
        if i > 0:
            y.grad.zero_()
        # print("==========y.grad: ", y.grad)
        z = torch.matmul(y, x) + b # linear layer    
        output = torch.sigmoid(z)
        label = torch.Tensor([0, 0, 1, 0, 0])
        loss = (output-label).var() # l2 loss
        loss.backward()
        
        # tensor a : requires_grad,  --> a.sub_(b): 对它的数据进行了更新；
        # pytorch check： 对我们需要更新梯度的tensor 禁止用 replace操作；
        # torch.no_grad(): 忽略这些警告，运行 replace 操作；
        with torch.no_grad(): # replace 
            y.sub_(0.2 * y.grad)
    
        print("loss: ", loss)
    
def internal_grad_demo():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True)    
    z = torch.matmul(x, w)+b # 全连接层
    o = z.sigmoid()
    print("z shape: ", z.shape)
    output_grad = torch.ones_like(o) # shape 和我正向传播时候的 output 的shape一致；
    o.backward(output_grad)
    print("w grad: ", w.grad) 
   
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
    
# 自定义方向函数
class Exp(torch.autograd.Function): #继承这个 function
    @staticmethod
    def forward(ctx, i): # context
        print("===========exp forward")
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output): # 自定义的方向传播函数
        print("============== exp backward")
        result, = ctx.saved_tensors
        return grad_output * result
    
def custom_demo():
    input = torch.randn(5, 6, requires_grad=True)
    output = Exp.apply(input) # output shape : (5, 6)
    output.backward(torch.rand(5, 6)) # 
    # print("output: ", input)
    # print("output grad: ", input.grad)
    
def outer(f):
    def inner(*args, **kargs):
        inner.co += 1
        print("=======: ", inner.co)
        return f(*args, **kargs)
    print("========== aaa")
    inner.co = 0
    return inner
    
def matmul_demo():
    input = torch.randn(10, 3, 4)
    mat2 = torch.randn(5, 1, 4, 5)
    # res = torch.bmm(input, mat2)
    # res = torch.mm(input, mat2)
    res = torch.matmul(input, mat2)
    res.is_leaf
    print(res)
    
def autograd_demo():
    data0 = torch.randn(2, 2, 4)
    w_0 = torch.randn(2, 4, 3, requires_grad=True)
    data2 = torch.bmm(data0, w_0)
    data3 = torch.sigmoid(data2)
    w_1 = torch.randn(2, 3, 5, requires_grad = True)
    output = torch.matmul(data3, w_1)
    # output.backward()
    output.backward(torch.ones_like(output))
    
    w_0 = w_0 - 0.001* w_0.grad

    print("run autograd_demo finished !!!")
          
if __name__ == "__main__":
    autograd_demo()
    # matmul_demo()
    # reqiregrad_set()
    # autograd_demo_v2()
    # internal_grad_demo()
    # set_no_grad()
    # grad_sum()
    # hook_demo()
    # get_inter_grad()
    custom_demo()
    print("run autograd_demo.py successfully !!!")