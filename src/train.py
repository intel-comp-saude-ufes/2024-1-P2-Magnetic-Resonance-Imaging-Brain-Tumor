from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter


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


def load_checkpoint(filename, model, optimizer, device, **kwargs):
    if not os.path.exists(filename):
        return None

    checkpoint = torch.load(filename, map_location=device)

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = checkpoint["criterion"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return epoch, model, criterion, optimizer


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

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                ## Statistics
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

                if save_test:
                    y_true.append(labels.cpu())
                    y_pred.append(outputs.cpu())
                    paths.extend(path)

                tqdm_eval.set_description(f"[ Testing ]" f"[ Loss: {running_loss/total:.6f} ]")

    avg_loss = running_loss / len(data_loader.dataset)

    result = None
    if save_test:
        result = OrderedDict(
            [
                ("y_true", torch.cat(y_true)),
                ("y_pred", torch.cat(y_pred)),
                ("paths", paths),
            ]
        )

    return avg_loss, result


def train_model(
    model,
    train_loader,
    val_loader,
    max_epochs: int,
    optimizer,
    criterion,
    device,
    checkpoint_dir="./checkpoints",
    resume_from: str = None,
    fold: str = None,
    writer: SummaryWriter = None,
    **kwargs,
):
    best_loss = np.inf

    epoch = 0
    if resume_from:
        epoch, model, criterion, optimizer = load_checkpoint(resume_from, model, optimizer, device)

    for epoch in range(epoch, max_epochs):
        loss = train_epoch(model, train_loader, epoch, max_epochs, criterion, optimizer, device)
        loss_val, _ = evaluate_model(model, val_loader, criterion, device)

        if writer:
            temp = {"train": loss, "val": loss_val}
            writer.add_scalars(f"loss_{fold}", temp, epoch + 1)

        filename = os.path.join(checkpoint_dir, "last_checkpoint.pth")
        save_checkpoint(filename, epoch + 1, model, criterion, optimizer, loss, loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            filename = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            save_checkpoint(filename, epoch + 1, model, criterion, optimizer, loss, loss_val)


def test_model(
    test_model,
    test_loader,
    criterion,
    device,
    test_dir="./tests/",
    **kwargs,
):
    _, result = evaluate_model(test_model, test_loader, criterion, device, save_test=True)
    filename = os.path.join(test_dir, "test_predictions.pth")
    torch.save(result, filename)
