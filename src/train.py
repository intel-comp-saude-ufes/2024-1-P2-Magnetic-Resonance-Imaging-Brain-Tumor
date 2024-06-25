from tqdm import tqdm
import numpy as np
import torch
import os
import pandas as pd


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


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = checkpoint["criterion"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return epoch, model, criterion, optimizer


def train_epoch(model, epoch, max_epoch, criterion, optimizer, data_loader, device):
    if device:
        model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(data_loader)) as pbar:
        for i, (inputs, labels, _) in enumerate(data_loader, 1):
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
                f"[ Training ]"
                f"[ Epoch: {epoch+1:02d}/{max_epoch:02d}, "
                f"Loss: {running_loss/i:.6f}, "
                f"Accuracy: {correct/total*100:.2f}% ]"
            )
            pbar.update()

    avg_loss = running_loss / len(data_loader)
    return avg_loss


def evaluate_model(model, criterion, data_loader, device, save_path=False):
    if device:
        model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []
    paths = []

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for i, (inputs, labels, path) in enumerate(data_loader, 1):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                ## statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.append(labels.cpu().numpy())
                y_pred.append(predicted.cpu().numpy())
                paths.append(path)

                pbar.set_description(
                    f"[ Testing ]"
                    f"[ Loss: {running_loss/i:.6f}, "
                    f"Accuracy: {correct/total*100:.2f}% ]"
                )
                pbar.update()

    avg_loss = running_loss / len(data_loader)
    if save_path:
        return avg_loss, (np.concatenate(y_true), np.concatenate(y_pred), paths)
    return avg_loss, (np.concatenate(y_true), np.concatenate(y_pred))


def train_model(
    model,
    max_epochs,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device,
    epoch=None,
    checkpoint_dir="./checkpoints",
):
    best_loss = np.inf
    optimizer = optimizer(model.parameters(), lr=0.001)

    for epoch in range(epoch or 0, max_epochs):
        loss = train_epoch(
            model, epoch, max_epochs, criterion, optimizer, train_loader, device
        )
        loss_val, _ = evaluate_model(model, criterion, val_loader, device)

        filename = os.path.join(checkpoint_dir, "last_checkpoint.pth")
        save_checkpoint(
            filename, epoch + 1, model, criterion, optimizer, loss, loss_val
        )

        if loss < best_loss:
            best_loss = loss
            filename = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            save_checkpoint(
                filename, epoch + 1, model, criterion, optimizer, loss, loss_val
            )


def test_model(model, criterion, test_loader, device):
    _, eval = evaluate_model(model, criterion, test_loader, device, save_path=True)

    paths = [x for i in eval[2] for x in i]
    df = pd.DataFrame()
    df["img"] = paths
    df["label"] = eval[0]
    df["prediction"] = eval[1]

    df.to_csv("test_predictions.csv", index=False)
