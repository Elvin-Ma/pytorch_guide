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
    label = torch.Tensor([0, 0, 1, 0, 0])
    
    grad_list = []
    
    def hook(grad):
        grad = grad * 2
        grad_list.append(grad)
        return grad # 可以直接更改里面存的梯度值的

    for i in range(100):
        # w.requires_grad=True # True 
        if w.grad:
          w.grad.zero_()
          
        z = torch.matmul(w, x) + b # linear layer    
        output = torch.sigmoid(z)
        output.register_hook(hook)        
        output.retain_grad() # tensor([-0.0405, -0.0722, -0.1572,  0.3101, -0.0403]
        loss = (output-label).var() # l2 loss
        loss.backward()
        print(w.grad)
        # # loss.backward()
        # output.backward(torch.ones_like(output))
        # print(w.grad)
        
        # grad --> 不存储的
        # w = w - 0.2 * grad_list[-1] # why wrong? --> 我们不能改变非叶子节点的requires_grad --> 
        # w1 = w.detach() # detach 
        # w = w1 - 0.2*grad_list[-1] # w 新建的tensor    
        print("loss: ", loss)

def autograd_demo_v2():
    torch.manual_seed(0)
    x = torch.ones(5, requires_grad=True)
    w = torch.randn(5, 5, requires_grad=True) # 叶子节点
    b = torch.randn_like(x)
    label = torch.Tensor([0, 0, 1, 0, 0])

    for i in range(100):
        # w.grad.zero_()
        w.retain_grad()
        # print("===========w.grad: ", w.grad)
        z = torch.matmul(w, x) + b # linear layer    
        output = torch.sigmoid(z)
        loss = (output-label).var() # l2 loss
        loss.backward()
        w = w - 0.2 * w.grad # y 新的y --> 已经完成了梯度清0；
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
    
    # z.backward(torch.ones_like(z))
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
    
def bp_demo():    
    w1 = [[0.1, 0.15], [0.2, 0.25], [0.3, 0.35]]
    w2 = [[0.4, 0.45, 0.5], [0.55, 0.6, 0.65]]
    w1 = torch.tensor(w1, requires_grad=True)
    w2 = torch.tensor(w2, requires_grad=True)
    b1 = torch.ones(3, 1).float()
    b2 = torch.ones(2, 2).float()
    
    input1 = torch.tensor([5, 10]).reshape(2, 1).to(torch.float32)
    label = torch.tensor([0.01, 0.99]).reshape(2, 1)
    
    for i in range(300):
        if w1.grad or w2.grad:
            w1.grad.zero_()
            w2.grad.zero_()
        
        w1.retain_grad()
        w2.retain_grad()
        net_h = torch.mm(w1, input1) + b1
        out_h = torch.sigmoid(net_h)

        net_o = torch.matmul(w2, out_h) + b2
        out_o = torch.sigmoid(net_o)
        loss = (out_o - label).var()
        loss.backward()
        
        if i < 100:
            w1 = w1 - 0.5 * w1.grad
            w2 = w2 - 0.5 * w2.grad
        else:
            w1 = w1 - 0.02 * w1.grad
            w2 = w2 - 0.02 * w2.grad

        print(loss)
        
def nn_demo():
    '''
    1. 数据准备：输入数据 + lable 数据
    2. 网络结构的搭建：激活函数 + 损失函数 + 权重初始化；
    3. 优化器选择；
    4. 训练策略：学习率的控制 + 梯度清0 + 更新权重 + 正则化；
    '''
    input = torch.tensor([5, 10]).reshape(1, 2).to(torch.float32)
    linear_1 = torch.nn.Linear(2, 3)
    act_1 = torch.nn.Sigmoid()
    linear_2 = torch.nn.Linear(3, 2)
    act_2 = torch.nn.Sigmoid()
    criteration = torch.nn.MSELoss()
    
    optimizer = torch.optim.SGD([{"params": linear_1.parameters()},
                                 {"params": linear_2.parameters()}], lr=0.5)
    label = torch.tensor([0.01, 0.99]).reshape(1, 2)
    
    for i in range(100):
        optimizer.zero_grad()
        x = linear_1(input)
        x = act_1(x)
        x = linear_2(x)
        output = act_2(x)
        loss = criteration(output, label)
        loss.backward()
        optimizer.step() # 更新权重      
        print(loss)
        
class ModuleDemo(torch.nn.Module):
    def __init__(self):
        super(ModuleDemo, self).__init__()
        self.linear_1 = torch.nn.Linear(2, 3)
        self.act_1 = torch.nn.Sigmoid()
        self.linear_2 = torch.nn.Linear(3, 2)
        self.act_2 = torch.nn.Sigmoid()
        
    def forward(self, input):
        x = self.linear_1(input)
        x = self.act_1(x)
        x = self.linear_2(x)
        output = self.act_2(x)
        # loss = self.criteration(output, label)
        return output
    
def module_train():
    input = torch.tensor([5, 10]).reshape(1, 2).to(torch.float32)
    label = torch.tensor([0.01, 0.99]).reshape(1, 2)
    model = ModuleDemo()
    criteration = torch.nn.MSELoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    
    for i in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criteration(output, label)
        loss.backward()
        optimizer.step()
        print(f"=========loss {loss}")
    
if __name__ == "__main__":
    # autograd_demo()
    # matmul_demo()
    # reqiregrad_set()
    # autograd_demo_v2()
    # internal_grad_demo()
    # set_no_grad()
    # grad_sum()
    # hook_demo()
    # get_inter_grad()
    # custom_demo()
    # autograd_demo_v2()
    # bp_demo()
    # nn_demo()
    module_train()
    print("run autograd_demo.py successfully !!!")