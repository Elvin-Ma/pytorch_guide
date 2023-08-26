import torch
import torch.nn as nn

def elu_demo():
    m = nn.ELU()
    input = torch.randn(2)
    output = m(input)
    print(output)
    
if __name__ == "__main__":
    elu_demo()
    print("run successfully")