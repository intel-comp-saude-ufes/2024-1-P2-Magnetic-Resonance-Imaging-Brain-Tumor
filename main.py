import torch
import pandas as pd
from src.train import train_model, test_model, load_checkpoint
from src.model import CNN
from src.datasets import split_datasets
from src.metrics import SoftDiceLoss
from src.utils import parse_arguments, initialize_wandb
from torch.utils.data import DataLoader


def get_dataloaders(batch_size, segmentation, multilabel, **kwargs):
    metadata = pd.read_csv("./dataset/data/info.csv")
    # TODO: mudar script de download para que o 'info.csv' inclua 'dataset/' no caminho
    metadata["image_path"] = "dataset/" + metadata["image_path"]
    metadata["mask_path"] = "dataset/" + metadata["mask_path"]

    # TODO: investigar se colocar exemplos sem tumor na segmentação faz diferença

    if segmentation:
        if multilabel:
            # class_dict = {"no_tumor": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}
            metadata = metadata[metadata["label"] != "no_tumor"]
            class_dict = {"glioma": 0, "meningioma": 1, "pituitary": 2}
        else:
            class_dict = {"no_tumor": 0, "glioma": 1, "meningioma": 1, "pituitary": 1}
    else:
        class_dict = {"no_tumor": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}

    metadata["label"] = metadata["label"].map(class_dict)

    training, test, validation = split_datasets(metadata, random_state=10, segmentation=segmentation, n_classes=metadata["label"].nunique())
    train_loader = DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    test_loader = test
    val_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)

    return (train_loader, test_loader, val_loader)


def run(configs):
    train, test, val = get_dataloaders(**configs)

    train_model(train_loader=train, val_loader=val, **configs)

    model = load_checkpoint("./checkpoints/best_checkpoint.pth", **configs)[1]
    test_model(model, test, **configs)


def classification_config():
    model = CNN("resnet101", 4)

    return dict(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )


def segmentation_config(multilabel):
    model = CNN("fcn_resnet101", 3 if multilabel else 1)

    return dict(
        model=model,
        criterion=SoftDiceLoss(multilabel),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )


if __name__ == "__main__":
    config = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.update({"device": device})

    # initialize_wandb(config)

    run_configs = segmentation_config(config["multilabel"]) if config["segmentation"] else classification_config()
    run_configs.update(config)

    run(run_configs)
