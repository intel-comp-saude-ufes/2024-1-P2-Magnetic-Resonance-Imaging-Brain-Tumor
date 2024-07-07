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

    last_checkpoint = configs["resume"]
    best_checkpoint = configs["test"]

    if last_checkpoint:
        date_time, fold = last_checkpoint.split("/")
        last_checkpoint = os.path.join("./runs", last_checkpoint, "last_checkpoint.pth")
    elif best_checkpoint:
        date_time, fold = best_checkpoint.split("/")
    else:
        date_time = datetime.now().strftime("run__%m_%d_%Y__%H_%M_%S")

    save_dir = os.path.join("./runs", date_time)
    if configs["tensor_board"]:
        writer_dir = os.path.join(save_dir, "tensorboard")
        writer = SummaryWriter(writer_dir)
    else:
        writer = None

    dataset = prepare_dataset(**configs)
    for i, splits in dataset:
        if last_checkpoint or best_checkpoint:
            if fold != i:
                continue

        configs.update(segmentation_config(configs["multilabel"], configs["device"]))

        fold_dir = os.path.join(save_dir, i)
        os.makedirs(fold_dir, exist_ok=True)

        train, test, val = prepare_dataloader(splits, configs["batch_size"])

        if not best_checkpoint:
            train_model(train_loader=train, val_loader=val, checkpoint_dir=fold_dir, resume_from=last_checkpoint, writer=writer, fold=i, **configs)
            if last_checkpoint:
                last_checkpoint = None

        best_checkpoint = os.path.join(fold_dir, "best_checkpoint.pth")
        model = load_checkpoint(best_checkpoint, **configs)[1]

        test_model(model, test, test_dir=fold_dir, **configs)
        best_checkpoint = None

    if configs["tensor_board"]:
        writer.close()


def segmentation_config(multilabel, device):
    model = CNN(3 if multilabel else 1).to(device)

    criterion = ComposedLoss(loss_funcs=[SoftDiceLoss(multilabel=multilabel), torch.nn.CrossEntropyLoss() if multilabel else torch.nn.BCEWithLogitsLoss()])

    return dict(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )


def printProblem(config):
    device = config["device"]
    print(f">> Running on {device}")

    model = type(config["model"].model).__name__
    if config["multilabel"]:
        print(f">> Starting {model} for multilabel segmentaion task...\n")
    else:
        print(f">> Starting {model} for segmentation task...\n")


if __name__ == "__main__":
    config = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.update({"device": device})

    run_configs = segmentation_config(config["multilabel"], device)
    run_configs.update(config)

    printProblem(run_configs)
    run(run_configs)
