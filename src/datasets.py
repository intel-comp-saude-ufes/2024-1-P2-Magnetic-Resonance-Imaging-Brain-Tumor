from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
import torch
from torchvision import transforms

data_transforms = transforms.Compose([  
    # transforms.ToPILImage(),
    # transforms.Resize([224,224]),
    # transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])
class BrainTumorDataset(Dataset):
    def __init__(self, data, transform=data_transforms):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["Brain_Image"]
        img = read_image(img_path, ImageReadMode.RGB)
        label = self.data.iloc[idx]["Tumor"]
        if self.transform:
            img = self.transform(img)
        img = img.to(dtype=torch.float32)
        return img, label, img_path
   


def getDataloaders(training_data, test_data, batch_size, shuffle=True):
    training_dataset = BrainTumorDataset(training_data)
    test_dataset = BrainTumorDataset(test_data)

    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=shuffle
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader


