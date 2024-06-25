from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.io import read_image


def getDataloaders(training_data, test_data, batch_size, shuffle=True):
    training_data = BrainTumorDataset(training_data)
    test_data = BrainTumorDataset(test_data)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, test_dataloader

class BrainTumorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data        
        self.transform = transform       
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['Brain_Image']
        img = read_image(img_path)
        label = self.data.iloc[idx]['Tumor']
        if self.transform:
            img = self.transform(img)
        return img, label

def train_epoch(model, epoch, max_epoch, criterion, optimizer, data_loader, device):
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(data_loader)) as pbar:
        for i, (inputs, labels) in enumerate(data_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            ## statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_description(
                f'[ Training ]'
                f'[ Epoch: {epoch+1:02d}/{max_epoch:02d}, '
                f'Loss: {running_loss/i:.6f}, '
                f'Accuracy: {correct/total*100:.2f}% ]'
            )
            pbar.update()

    return model

def evaluate_model(model, criterion, data_loader, device):
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for i, (inputs, labels) in enumerate(data_loader, 1):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                ## statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels)
                y_pred.extend(predicted)

                pbar.set_description(
                    f'[ Testing ]'
                    f'[ Loss: {running_loss/i:.6f}, '
                    f'Accuracy: {correct/total*100:.2f}% ]'
                )
                pbar.update()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true, y_pred


def train_model(model, criterion, optimizer, train_loader, device, max_epochs, val_loader=None):
    for epoch in range(max_epochs):
        model = train_epoch(model, epoch, max_epochs, criterion, optimizer, train_loader, device)

        # TODO: save checkpoints

        if val_loader is not None:
            y_true, y_pred = evaluate_model(model, criterion, val_loader, device)

def test_model(model, criterion, test_loader, device):
    y_true, y_pred = evaluate_model(model, criterion, test_loader, device)

    # TODO: save test
