from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

class BrainTumorDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.data = np.array(img_paths)
        self.labels = np.array(labels)
      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")              
        label = self.labels[idx]
        img = self.transform(img)    
        
        return img, label, img_path

transform_train = transforms.Compose([      
                transforms.ColorJitter(brightness=0.4, contrast=0.3),
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #discutir importância
            ])

transform_eval = transforms.Compose([             
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

from sklearn.model_selection import train_test_split

def getDatasets(training_path, test_path, random_state):
    training = pd.read_csv(training_path)
    train_imgs_paths, train_labels = training['Brain_Image'], training['Tumor']
    testing = pd.read_csv(test_path)
    temp_imgs_paths, temp_labels = testing['Brain_Image'], testing['Tumor']

    test_imgs_paths, val_imgs_paths, test_labels, val_labels = train_test_split(
                                                                temp_imgs_paths, temp_labels, test_size=0.5, 
                                                                random_state=10)
    
    train_dataset = BrainTumorDataset(train_imgs_paths, train_labels, transform=transform_train)
    test_dataset = BrainTumorDataset(test_imgs_paths, test_labels, transform=transform_eval)
    val_dataset = BrainTumorDataset(val_imgs_paths, val_labels, transform=transform_eval)
    
    return train_dataset, test_dataset, val_dataset


    
