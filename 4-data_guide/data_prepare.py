from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pandas as pd
import os
import pillow as PIL
from torchvision.io import read_image

def data_download():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
# 自己的一个dataset ： 客户自己的dataset
# 作用：方便我们处理数据；加速我们处理数据
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # 每一张图片名称，以及对应的label
        self.img_dir = img_dir # 图像的跟目录
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    # 最核心之处：idx
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 得到一张图片的完整路径
        image = read_image(img_path) # 用我们的图像读取工具来读取图片（opencv、pillow）
        label = self.img_labels.iloc[idx, 1] # 读取图片对应的label
        if self.transform:
            image = self.transform(image) # 图像的预处理
        if self.target_transform:
            label = self.target_transform(label) # 标签的预处理
        return image, label #把最终结果返回
    
def data_loader():
    from torch.utils.data import DataLoader
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    # )
    
    train_data = CustomImageDataset() # 实例化dataset
    
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # batch_idx : 第几次循环
        # data：最终输入data
        # target：label
        pass
        
if __name__ == "__main__":
    data_download()
    print("run data_prepare.py successfully !!!")
    


