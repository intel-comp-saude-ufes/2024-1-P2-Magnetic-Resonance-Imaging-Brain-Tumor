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

        ''' TODO
        preciso de duas variaveis pra fazer o kfold
        self.data (lista/array/etc de paths das imagens)
        self.labels (lista/array/etc labels das imagens)
        '''

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["Brain_Image"]
        img = read_image(img_path, ImageReadMode.RGB)

        ''' TODO?
        esse read_image ja carrega a imagem como tensor,
        então na hora do transforms.ToTensor() ta bugando,
        mas desse jeito o range fica entre [0, 255] talvez
        seja melhor ler com a biblioteca PIL e fazer o
        transform, que fica entre [0, 1]
        '''

        label = self.data.iloc[idx]["Tumor"]
        if self.transform:
            img = self.transform(img)
        img = img.to(dtype=torch.float32)
        return img, label, img_path


def load_dataset(dataset_path):
    ''' TODO
    pode carregar o dataset todo e retornar
    '''

    return BrainTumorDataset()
