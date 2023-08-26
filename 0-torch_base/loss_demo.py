import torch
import torch.nn as nn
import torch.nn.functional as F

def l1_loss_demo():
    loss = nn.L1Loss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input, target)
    output.backward()
    
def mse_loss_demo():
    loss = nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input, target)
    output.backward()
    
    
def bce_loss_demo():
    m = nn.Sigmoid()
    loss = nn.BCELoss() # 
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    output.backward()
    
def cross_entropy_demo():
    # 定义模型输出和目标标签
    model_output = torch.tensor([[0.1, 0.2, 0.3, 0.3], [0.4, 0.5, 0.6, 0.8]])
    target_labels = torch.tensor([3, 0])
    # 创建交叉熵损失函数实例
    criterion = nn.CrossEntropyLoss()
    # 计算损失
    loss = criterion(model_output, target_labels)
    print(loss)
    
def nllloss_demo():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output.backward()
    # 2D loss example (used, for example, with image inputs)
    N, C = 5, 4
    loss = nn.NLLLoss()
    # input is of size N x C x height x width
    data = torch.randn(N, 16, 10, 10)
    conv = nn.Conv2d(16, C, (3, 3))
    m = nn.LogSoftmax(dim=1)
    # each element in target has to have 0 <= value < C
    target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    output = loss(m(conv(data)), target)
    output.backward()
    
def dice_loss_demo():
    def dice_loss(output, target, smooth=1e-6):
        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score
        return dice_loss

    # 定义模型输出和目标标签
    model_output = torch.tensor([[0.8, 0.2, 0.3], [0.4, 0.5, 0.6]])
    target_labels = torch.tensor([[1, 0, 1], [0, 1, 1]])
    # 计算Dice Loss
    loss = dice_loss(model_output, target_labels)
    print(loss)
    
def iou_loss_demo():
    def iou_loss(pred_boxes, target_boxes, smooth=1e-6):
        # 计算候选框的坐标
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        # 计算候选框的面积
        pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
        target_area = (target_x2 - target_x1 + 1) * (target_y2 - target_y1 + 1)
        # 计算交集的坐标
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        # 计算交集的面积
        inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)
        # 计算并集的面积
        union_area = pred_area + target_area - inter_area
        # 计算IoU
        iou = (inter_area + smooth) / (union_area + smooth)
        # 计算IoU Loss
        iou_loss = 1.0 - iou.mean()
        return iou_loss

    # 定义模型输出的候选框和目标候选框
    model_boxes = torch.tensor([[10, 10, 100, 100], [20, 20, 120, 120]])
    target_boxes = torch.tensor([[15, 15, 105, 105], [30, 30, 110, 110]])
    # 计算IoU Loss
    loss = iou_loss(model_boxes, target_boxes)
    print(loss)
 
def focal_loss():
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                focal_loss = torch.mean(focal_loss)
            elif self.reduction == 'sum':
                focal_loss = torch.sum(focal_loss)

            return focal_loss
        
    # 创建模型实例
    model = MyModel()
    # 创建Focal Loss实例
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    # 定义输入数据和目标标签
    input_data = torch.randn(1, 10)
    targets = torch.tensor([0])
    # 前向传播
    output = model(input_data)
    # 计算损失
    loss = focal_loss(output, targets)
    # 反向传播
    loss.backward()
 
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def smooth_l1_loss_demo():
    # 创建模型
        
    # 创建Smooth L1 Loss实例
    smooth_l1_loss = nn.SmoothL1Loss()
    # 创建模型实例
    model = MyModel()
    # 定义输入数据和目标标签
    input_data = torch.randn(1, 10)
    targets = torch.randn(1, 1)
    # 前向传播
    output = model(input_data)
    # 计算损失
    loss = smooth_l1_loss(output, targets)
    # 反向传播
    loss.backward()
    
if __name__ == "__main__":
    # l1_loss_demo()
    # cross_entropy_demo()
    # iou_loss_demo()
    # bce_loss_demo()
    # focal_loss()
    dice_loss_demo()
    print("run loss_demo.py successfully !!!")