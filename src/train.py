from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os


def save_checkpoint(filename, epoch, model, criterion, optimizer, loss, loss_val):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "criterion": criterion,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "loss_val": loss_val,
    }

    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, **kwargs):
    checkpoint = torch.load(filename)

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = checkpoint["criterion"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return epoch, model, criterion, optimizer


def save_state(self, fold, epoch, model, optimizer, schedulers, file_name=None):
    if file_name is None:
        file_name = "model_fold_{:02d}_epochs_{:04d}.pt".format(fold, epoch)

    train_state_path = os.path.join(self.models_dirpath, file_name)
    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **{f"scheduler{i}": scheduler.state_dict() for i, scheduler in enumerate(schedulers)},
        },
        train_state_path,
    )


def train_epoch(model, data_loader, epoch, max_epoch, criterion, optimizer, device):
    model.to(device)
    model.train()

    running_loss = 0.0
    total = 0

    with tqdm(data_loader) as tqdm_train:
        for inputs, labels, _ in tqdm_train:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            ## Statistics
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            tqdm_train.set_description(f"[ Training ]" f"[ Epoch: {epoch+1:02d}/{max_epoch:02d}, " f"Loss: {running_loss/total:.6f} ]")

    avg_loss = running_loss / len(data_loader.dataset)
    return avg_loss


def evaluate_model(model, data_loader, criterion, device, save_test=False):
    model.to(device)
    model.eval()

    running_loss = 0.0
    total = 0

    y_true = []
    y_pred = []
    paths = []

    with torch.no_grad():
        with tqdm(data_loader) as tqdm_eval:
            for inputs, labels, path in tqdm_eval:

                inputs, labels = inputs.to(device), labels.to(device)
                if len(inputs.shape) < 4:
                    inputs = inputs.unsqueeze(0)

                # Forward pass
                outputs = model(inputs)
                if outputs.shape[0] == 1:
                    outputs = outputs.squeeze()

                loss = criterion(outputs, labels)

                ## Statistics
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

                if save_test:
                    if len(labels.shape) < 1:
                        y_true.append(labels.cpu().item())
                    else:
                        y_true.append(labels.cpu())
                    y_pred.append(outputs.cpu())
                    paths.append(path)

                tqdm_eval.set_description(f"[ Testing ]" f"[ Loss: {running_loss/total:.6f} ]")

    avg_loss = running_loss / len(data_loader)

    result = None
    if save_test:
        result = OrderedDict(
            [
                ("y_true", torch.Tensor(y_true) if len(labels.shape) < 1 else torch.cat(y_true)),
                ("y_pred", torch.cat(y_pred)),
                ("paths", paths),
            ]
        )

    return avg_loss, result


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int, criterion, optimizer, device, epoch=None, checkpoint_dir="./checkpoints", **kwargs):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    best_loss = np.inf
    history = defaultdict(list)

    epoch = epoch if isinstance(epoch, int) else 0
    for epoch in range(epoch, max_epochs):
        loss = train_epoch(model, train_loader, epoch, max_epochs, criterion, optimizer, device)
        loss_val, _ = evaluate_model(model, val_loader, criterion, device)

        history["loss"].append(loss)
        history["loss_val"].append(loss_val)
        history["epoch"].append(epoch)

        filename = os.path.join(checkpoint_dir, "last_checkpoint.pth")
        save_checkpoint(filename, epoch + 1, model, criterion, optimizer, loss, loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            filename = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            save_checkpoint(filename, epoch + 1, model, criterion, optimizer, loss, loss_val)

    return history


def test_model(test_model, test_loader, criterion, device, test_dir="./tests", **kwargs):
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for fold, dataloader in test_loader:
        _, result = evaluate_model(test_model, dataloader, criterion, device, save_test=True)
        torch.save(result, test_dir + f"/{fold}_predictions.pth")
