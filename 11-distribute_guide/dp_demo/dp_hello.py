import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        # self.modules = [self.fc, self.sigmoid]

    def forward(self, input):
        return self.sigmoid(self.fc(input))


if __name__ == '__main__':
    # Parameters and DataLoaders
    input_size = 5
    output_size = 1
    batch_size = 30
    data_size = 100

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
    cls_criterion = nn.BCELoss()

    for data in rand_loader:
        targets = torch.empty(data.size(0)).random_(2).view(-1, 1)

        if torch.cuda.is_available():
            input = Variable(data.cuda())
            with torch.no_grad():
                targets = Variable(targets.cuda())
        else:
            input = Variable(data)
            with torch.no_grad():
                targets = Variable(targets)

        output = model(input)

        optimizer.zero_grad()
        loss = cls_criterion(output, targets)
        loss.backward()
        optimizer.step()
