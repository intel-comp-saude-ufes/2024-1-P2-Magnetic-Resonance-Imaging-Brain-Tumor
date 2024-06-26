import pandas as pd
import torch
from src.train import train_model, test_model
from src.model import CNN
from src.datasets import getDataloaders, get_mean_std




if __name__ == "__main__":
    model = CNN("vgg16", 4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam

    train_df = pd.read_csv("processed_data/training_data.csv")
    test_df = pd.read_csv("processed_data/testing_data.csv")
    train, test = getDataloaders(train_df, test_df, 256)

    train_model(model, 2, criterion, optimizer, train, test, "cuda")
    test_model(model, criterion, test, "cuda")
