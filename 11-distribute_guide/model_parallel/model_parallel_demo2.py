import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000

class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
    
class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0)) # 对batch 维度进行split
        s_next = next(splits)
        s_prev = self.seq1(s_next) # s_prev --> cuda:0
        s_prev = s_prev.to('cuda:1') # s_prev --> cuda:0
        ret = []

        for s_next in splits:
            # A. ``s_prev`` runs on ``cuda:1``
            s_prev = self.seq2(s_prev) # s_prev: cuda:1
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. ``s_next`` runs on ``cuda:0``, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1') # 执行seq1

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


# setup = "model = PipelineParallelResNet50()"
# pp_run_times = timeit.repeat(
#     stmt, setup, number=1, repeat=num_repeat, globals=globals())
# pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

# plot([mp_mean, rn_mean, pp_mean],
#      [mp_std, rn_std, pp_std],
#      ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
#      'mp_vs_rn_vs_pp.png')    
    
# import torchvision.models as models

# num_batches = 3
# batch_size = 120
# image_w = 128
# image_h = 128


# def train(model):
#     model.train(True)
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001)

#     one_hot_indices = torch.LongTensor(batch_size) \
#                            .random_(0, num_classes) \
#                            .view(batch_size, 1)

#     for _ in range(num_batches):
#         # generate random inputs and labels
#         inputs = torch.randn(batch_size, 3, image_w, image_h)
#         labels = torch.zeros(batch_size, num_classes) \
#                       .scatter_(1, one_hot_indices, 1)

#         # run forward pass
#         optimizer.zero_grad()
#         outputs = model(inputs.to('cuda:0'))

#         # run backward pass
#         labels = labels.to(outputs.device)
#         loss_fn(outputs, labels).backward()
#         optimizer.step()