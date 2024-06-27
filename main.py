import torch
from src.train import train_model, test_model, load_checkpoint
from src.model import CNN
from src.datasets import load_dataset

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("path/dataset")

    n_splits = 5
    max_epochs = 10
    batch_size = 32

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(dataset.data, dataset.labels), 1):
        print(f'Fold {fold}/{n_splits}')

        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, stratify=dataset.labels[train_val_idx])

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        model = CNN("resnet18", 4, freeze_conv=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        train_model(model, max_epochs, criterion, optimizer, train_loader, val_loader, device)
        model = load_checkpoint("./checkpoints/best_checkpoint.pth")[1]
        test_model(model, criterion, test_loader, device)
        print()


if __name__ == "__main__":
    main()
