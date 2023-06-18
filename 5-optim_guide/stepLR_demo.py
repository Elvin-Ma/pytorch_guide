import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim

def step_lr():
    model = nn.Linear(30, 10) # 以一个最简单的model
    input = torch.randn(4, 30)
    label = torch.Tensor([2, 4, 5, 6]).to(torch.int64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.01,
        momentum=0.9,
        weight_decay=0.01)
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)       
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.1, verbose=True)       
    
    for i in range(100):
        model.train()
        optimizer.zero_grad()
        output = torch.sigmoid(model(input))
        loss = criterion(output, label)
        loss.backward()
        print("==========step: ", i)
        optimizer.step()
        # if i % 100 == 0: # 我就认为 100 步完成一个 epoch
        scheduler.step()
    
if __name__ == "__main__":
    step_lr()
    print("run stepLR demo.py successfully !!!")