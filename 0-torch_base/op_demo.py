import copy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision

# n, cin, h, w, cout, kernel_size, stride, padding, dilation, eps
params = [
  [16, 32, 7, 7, 32, 3, 1, 1, 1, 1e-20]
]

def get_offset_mask(out, khmkw):
    offset = out[:,:2*khmkw]
    mask = out[:,2*khmkw:]
    mask = torch.sigmoid(mask)
    return offset, mask

def deform_conv2d_demo(n, cin, h, w, cout, kernel_size, stride, padding, dilation, eps):
  input_cpu = torch.randn(params[0][:4])
  conv_cpu = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
  conv_output_cpu = conv_cpu(input_cpu)

  conv_output_cpu.retain_grad()
  dc_m_cpu = torchvision.ops.DeformConv2d(cin, cout, kernel_size, stride, padding, dilation, groups=1, bias=False)

  conv_offset = nn.Conv2d(cin, 3 * kernel_size * kernel_size,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=False)

  offset_mask_cpu = conv_offset(input_cpu)
  offset_cpu, mask_cpu = get_offset_mask(offset_mask_cpu, kernel_size * kernel_size)
  dc_m_out_cpu = dc_m_cpu(conv_output_cpu, offset_cpu, mask_cpu)

  print(f"========input_cpu shape: {input_cpu.shape}")
  print(f"========conv_output_cpu: shape {conv_output_cpu.shape}")
  print(f"========offset_mask_cpu: shape {offset_mask_cpu.shape}")
  print(f"========offset_cpu shape: {offset_cpu.shape}")
  print(f"========mask_cpu shape: {mask_cpu.shape}")
  print(f"===========dc_m_out_cpu: {dc_m_out_cpu.shape}")


def transposed_conv_demo():
  in_channels = 16
  out_channels = 32
  kernel_size = 3
  stride = 2
  padding = 1

  # 创建转置卷积层
  trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

  # 生成一个随机输入张量
  batch_size = 1
  input_height = 8
  input_width = 8
  input_channels = in_channels
  input = torch.randn(batch_size, input_channels, input_height, input_width)

  # 进行转置卷积操作
  output = trans_conv(input)

  # 打印输出张量的形状
  print(output.shape)

def group_conv_demo():
  # 定义分组卷积的参数 # weight = [32, 16/4, 3, 3]
  in_channels = 16
  out_channels = 32
  kernel_size = 3
  stride = 1
  padding = 1
  groups = 16

  # 创建分组卷积层
  group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)# groups = 4
  # weight_shape = (groups)

  print("groups shape: ", groups)
  print("output channels: ", out_channels)

  # 生成一个随机输入张量
  batch_size = 1
  input_height = 8
  input_width = 8
  input_channels = in_channels
  input = torch.randn(batch_size, input_channels, input_height, input_width) # [1 , 16, 8, 8]
  print("input shape: ", input.shape)

  # 进行分组卷积操作
  output = group_conv(input)

  # 打印输出张量的形状
  print("output shape: ", output.shape)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
      

def batch_morm_demo():
  num_features = 16
  eps = 1e-5
  momentum = 0.1

  # 创建批归一化层
  batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
  # 定义层归一化的参数(laynorm)
  normalized_shape = [16, 8, 8]

  # 创建层归一化层
  layer_norm = nn.LayerNorm(normalized_shape)

  # 生成一个随机输入张量
  batch_size = 1
  input_height = 8
  input_width = 8
  input_channels = num_features
  input = torch.randn(batch_size, input_channels, input_height, input_width)

  # 进行批归一化操作
  batch_norm_output = batch_norm(input)

  # 进行层归一化操作
  layer_norm_output = layer_norm(input)
  print(layer_norm_output.shape)


def conv_demo():
  # With square kernels and equal stride
  m = nn.Conv2d(16, 33, 3, stride=2, padding=1) # 初始化
  # # non-square kernels and unequal stride and with padding
  # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
  # # non-square kernels and unequal stride and with padding and dilation
  # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
  input = torch.randn(20, 16, 50, 100)
  output = m(input) # 真正的计算
  print("output shape", output.shape)
  
def linear_demo():
  m = nn.Linear(20, 30) # 初始化 [*,k]*[k,n] --> [*,n] --> weight bias : 自动帮我们做了初始化
  input = torch.randn(512, 20) # input 设定
  layer_0 = m(input) # run weight 是有一个转置的
  
  m_2 = nn.Linear(30, 40)
  output = m_2(layer_0)
  print(output.size()) # bias 有没有， weight呢？

def  maxpool_demo():
   m = nn.AvgPool2d(3, stride=2, padding=1)
   # pool of non-square window
  #  m = nn.MaxPool2d((3, 2), stride=(2, 1))
   input = torch.randn(20, 16, 50, 32)
   output = m(input)
   print("output shape: ", output.shape)
   
def global_average_pool():
  m = nn.AdaptiveAvgPool2d((2, 2))
  input = torch.randn(1, 2048, 7, 7)
  output = m(input)
  torch.onnx.export(m, input, "adaptiv_avg.onnx")
  
  print("output shape: ", output.shape)
  
torch.Tensor()
def batch_norm_demo():
  m = nn.BatchNorm2d(100) # 100 就表示我们的channel 维度
  # Without Learnable Parameters
  # m = nn.BatchNorm2d(100, affine=False)
  input = torch.randint(-100, 100, (20, 100, 35, 45)).float() # 哪些数据取均值（20*35*45） --> 取了多少个均值
  output = m(input)
  # input2 = torch.randint(-10, 200, (20, 100, 35, 45)).float()
  # output1 = m(input2)
  print("output shape: ", output.shape)
  
def rnncell_onnx_get():
  rnn = nn.RNNCell(10, 20)
  input = torch.randn(6, 3, 10)
  hx = torch.randn(3, 20)
  output = []
  torch.onnx.export(rnn, input[0], "rnn_cell.onnx")
  # for i in range(6):
  #     hx = rnn(input[i], hx)
  #     output.append(hx)
   
def rnn_onnx_get():
  rnn = nn.RNN(10, 20, 2)
  input = torch.randn(5, 3, 10)
  h0 = torch.randn(2, 3, 20)
  output, hn = rnn(input, h0)
  torch.onnx.export(rnn, (input, h0), "rnn.onnx")
  # model = onnx.shape_inference.infer_shapes(onnx_model)
  
def flatten_demo():
  input = torch.randn(32, 1, 5, 5)
  # With default parameters
  m = nn.Flatten()
  output = m(input)
  output.size()
  # h.Size([32, 25])
  # With non-default parameters
  m = nn.Flatten(0, 2)
  output = m(input)
  output.size()
  
def embedding_demo():
  # weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
  # embedding = nn.Embedding.from_pretrained(weight)
  # # Get embeddings for index 1
  # input = torch.LongTensor([1]) # 35200 --> [0, 1, ... , 35199]
  # output = embedding(input)
  # print("output: ", output)
  
  embedding = nn.Embedding(10, 3)
  # a batch of 2 samples of 4 indices each
  input = torch.LongTensor([[1, 2, 4, 9], [4, 3, 2, 9]]) # int64_t
  output = embedding(input)
  print(output.shape)
  
def gather_demo():
  t = torch.tensor([[1, 2], [3, 4]])
  output = torch.gather(t, 0, torch.tensor([[0, 0], [1, 0]]))
  print(output.shape)

class DropConnect(nn.Module):
    def __init__(self, p=0.5):
        super(DropConnect, self).__init__()
        self.p = p  # DropConnect 概率

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 生成 DropConnect 掩码
        mask = torch.empty_like(x).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)  # 缩放掩码，使期望值保持不变

        # 应用 DropConnect 掩码
        x = x * mask
        return x

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p  # Dropout 概率
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 生成 Dropout 掩码
        mask = torch.empty_like(x).bernoulli_(1 - self.p)
        if self.inplace:
            x.mul_(mask)
        else:
            x = x * mask

        return x

if __name__ == "__main__":
  gather_demo()
  # deform_conv2d_demo(*params[0])
  # transposed_conv_demo()
  # group_conv_demo()
  # conv_demo()
  # linear_demo()
  # maxpool_demo()
  # global_average_pool()
  # batch_norm_demo()  
  # batch_morm_demo()
  #rnn_onnx_get()
  # flatten_demo()
  embedding_demo()
  print("run op_demo.py successfully !!!")
