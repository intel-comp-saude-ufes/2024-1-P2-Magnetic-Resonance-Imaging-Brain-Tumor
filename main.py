import torch
import pandas as pd
from src.train import train_model, test_model, load_checkpoint
from src.model import CNN
from src.datasets import split_datasets, prepare_dataloader
from src.metrics import SoftDiceLoss, ComposedLoss
from src.utils import parse_arguments
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter


def prepare_dataset(multilabel, cv, test_size, val_size, **kwargs):
    metadata = pd.read_csv("./dataset/data/info.csv")
    metadata["label"] = metadata["label"].map({"glioma": 0, "meningioma": 1, "pituitary": 2}) if multilabel else 1

    return split_datasets(
        metadata,
        n_classes=(metadata["label"].nunique() if multilabel else 1),
        test_size=(test_size if not cv else cv),
        val_size=val_size,
        cv=cv is not None,
        random_state=10,
    )


def run(configs):
    now = datetime.now()
    date_time = now.strftime("run__%m_%d_%Y__%H_%M_%S")

    writer = SummaryWriter("runs_tensor") if configs["tensor_board"] else None

    dataset = prepare_dataset(**configs)
    for i, splits in dataset:
        save_dir = os.path.join("./runs", date_time, i)
        os.makedirs(save_dir)

        train, test, val = prepare_dataloader(splits, configs["batch_size"])

        train_model(train_loader=train, val_loader=val, checkpoint_dir=save_dir, writer=writer, **configs)

        best_checkpoint = os.path.join(save_dir, "best_checkpoint.pth")
        model = load_checkpoint(best_checkpoint, **configs)[1]

        test_model(model, test, test_dir=save_dir, **configs)

    if configs["tensor_board"]:
        writer.close()


def segmentation_config(multilabel):
    model = CNN(3 if multilabel else 1)

    criterion = ComposedLoss(loss_funcs=[SoftDiceLoss(multilabel=multilabel), torch.nn.CrossEntropyLoss() if multilabel else torch.nn.BCEWithLogitsLoss()])

    return dict(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )


def printProblem(config):
    device = config["device"]
    print(f">> Running on {device}")

    if config["multilabel"]:
        print(">> Starting ResNet model for multilabel segmentaion task...")
    else:
        print(">> Starting ResNet model for segmentation task...")

    print()


if __name__ == "__main__":
    config = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.update({"device": device})
    
    print(config)

    printProblem(config=config)

    run_configs = segmentation_config(config["multilabel"])
    run_configs.update(config)

    run(run_configs)
