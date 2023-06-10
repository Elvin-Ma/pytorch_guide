from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim


def step_lr():
    model = nn.Linear(30, 10)
    input = torch.randn(4, 30)
    label = torch.Tensor([2, 4, 5, 6]).to(torch.int64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.01,
        momentum=0.9,
        weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)       
    
    for i in range(1000):
        model.train()
        output = torch.sigmoid(model(input))
        loss = criterion(output, label)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        if i % 100 == 0:
            scheduler.step()
    
if __name__ == "__main__":
    step_lr()
    print("run stepLR demo.py successfully !!!")