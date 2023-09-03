from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
 
def save_demo_v1():
    model = Net()
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    torch.save(model, "mnist.pt") # 4.6M : 保存
    
def load_demo_v1():
    model = torch.load("mnist.pt")
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    print(f"output shape: {output.shape}")
    
def save_para_demo():
    model = Net()
    torch.save(model.state_dict(), "mnist_para.pth")
    
def load_para_demo():
    param = torch.load("mnist_para.pth")
    model = Net()
    model.load_state_dict(param)
    input = torch.rand(1, 1, 28, 28)
    output = model(input)   
    print(f"output shape: {output.shape}")

def tensor_save():
    tensor = torch.ones(5, 5)
    torch.save(tensor, "tensor.t")
    tensor_new = torch.load("tensor.t")
    print(tensor_new)
    
def load_to_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('mnist.pt', map_location=device)
    print(f"model device: {model}")

def save_ckpt_demo():
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = torch.Tensor([0.25])
    epoch = 10
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss.item(),
        # 可以添加其他训练信息
    }

    torch.save(checkpoint, 'mnist.ckpt')

def load_ckpt_demo():
    checkpoint = torch.load('model.ckpt')
    model = Net() # 需要事先定义一个net的实例
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    print("output shape: ", output.shape)
    
def save_trace_model():
    model = Net().eval()
    # 通过trace 得到了一个新的model，我们最终保存的是这个新的model
    traced_model = torch.jit.trace(model, torch.randn(1, 1, 28, 28))
    traced_model.save("traced_model.pt")
    # torch.save(traced_model, "mnist_trace.pt")
    
def load_trace_model():
    mm = torch.jit.load("traced_model.pt")
    output = mm(torch.randn(1, 1, 28, 28))
    print("load model succsessfully !")
    print("output: ", output)
        
if __name__ == "__main__":
    # save_demo_v1()
    # load_demo_v1()
    # save_para_demo()
    # load_para_demo()
    # tensor_save()
    # load_to_gpu()
    # save_trace_model()
    # load_trace_model()
    save_ckpt_demo()
    # load_ckpt_demo()
    print("run save_load_demo.py successfully !!!")
