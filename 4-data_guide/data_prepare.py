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
    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def data_loader():
    from torch.utils.data import DataLoader
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    # )
    train_data = CustomImageDataset()
    
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # batch_idx : 第几次循环
        # data：最终输入data
        # target：label
        pass
        
if __name__ == "__main__":
    data_download()
    print("run data_prepare.py successfully !!!")
    


