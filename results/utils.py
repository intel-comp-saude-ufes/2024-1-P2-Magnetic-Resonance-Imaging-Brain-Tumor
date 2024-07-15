import os

import pandas as pd
import numpy as np
import torch

from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image

from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(files):
    y_true, y_pred = [], []
    for _, fold in files.items():
        outputs = fold["y_pred"]
        img_paths = fold["paths"]
        _, labels = get_all_info(img_paths)

        y_true.append(torch.from_numpy(labels))
        y_pred.append(torch.argmax(outputs["class"], 1))

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    labels = [i.title() for i in ["glioma", "meningioma", "pituitary"]]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp = disp.plot(cmap="Blues")
    plt.yticks(rotation=90, ha='center', rotation_mode='anchor')
    plt.savefig("disp.svg")
    return disp


def _dice(output, target):
    dims = (-2, -1)
    smooth = 1e-6

    output = output.sigmoid()
    output = torch.where(output > 0.5, 1, 0)

    tp = (output * target).sum(dim=dims)
    fp = (output * (1 - target)).sum(dim=dims)
    fn = ((1 - output) * target).sum(dim=dims)
    return ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).mean()


def read_test(run_folder: str):
    tests = {}
    for fold in sorted(os.listdir(run_folder)):
        if fold.startswith("fold_"):
            test_predictions = os.path.join(run_folder, fold, "test_predictions.pth")
            tests[fold] = torch.load(test_predictions)
    return tests


resize = A.Resize(224, 224)


def show_image(y_true: torch.Tensor, y_pred: torch.Tensor, path, ax: Axes):
    img = np.array(Image.open("../" + path).convert("RGB").convert("L"))
    img = resize(image=img)
    ax.imshow(img["image"], cmap="gray")
    ax.imshow(y_pred.sigmoid(), cmap=ListedColormap(colors=[(0.0, 0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 1.0)]), alpha=0.3)
    ax.imshow(y_true, cmap=ListedColormap(colors=[(0.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 1.0)]), alpha=0.3)


info = pd.read_csv("../dataset/data/info.csv")
info["label"] = info["label"].map({"glioma": 0, "meningioma": 1, "pituitary": 2})


def show_patient_results(idx, outputs, masks, labels, img_paths, title):
    n_imgs = len(idx)
    cols = min(n_imgs, 6)
    rows = (n_imgs // (cols + 1)) + 1

    fig, axes = plt.subplots(rows, cols, sharey=True, figsize=(cols * 3, 3.1 * rows))
    fig.suptitle(title)

    if cols == 1:
        axes = np.array(axes)
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()

    idx = np.array(idx)
    for ax, seg, y_pred, mask, y_true, i in zip(axes, outputs["seg"][idx], outputs["class"][idx], masks[idx], labels[idx], idx):
        dice = _dice(seg, mask) * 100
        ax.set_title("{:02.02f}%".format(dice))
        show_image(mask, seg, img_paths[i], ax)


def calculate_patient_scores(outputs, masks, labels, img_paths):
    n_indices = len(outputs)
    available_indices = set(range(n_indices))
    p_scores = []

    while len(available_indices) != 0:
        pid_fold = available_indices.pop()
        p_img_ref = img_paths[pid_fold]
        p_idx, _, _ = get_patient_info(p_img_ref, img_paths)

        dice = _dice(outputs["seg"][p_idx], masks[p_idx])
        acc = recall_score(labels[p_idx], np.argmax(outputs["class"][p_idx], 1), average="macro")
        p_scores.append((p_idx, dice, acc))

        available_indices -= set(p_idx)

    return p_scores


def show(files):
    scores = {}
    for name, fold in files.items():
        outputs = fold["y_pred"]
        img_paths = fold["paths"]
        masks, labels = get_all_info(img_paths)

        dice = _dice(outputs["seg"], masks)

        y_pred = torch.argmax(outputs["class"], 1)
        acc = recall_score(labels, y_pred, average="macro")

        print(name)
        print("Dice (F1 macro): {:02.02f}%".format(dice * 100))
        print("Balanced Accuracy: {:02.02f}%".format(acc * 100))
        scores[name] = (dice, acc)

        p_scores = calculate_patient_scores(outputs, masks, labels, img_paths)
        p_scores.sort(key=lambda x: x[1])

        w_idx, w_score, _ = p_scores[0]
        show_patient_results(w_idx, outputs, masks, labels, img_paths, name + " worst dice score: {:02.02f}".format(w_score))

        b_idx, b_score, _ = p_scores[-1]
        show_patient_results(b_idx, outputs, masks, labels, img_paths, name + " best dice score: {:02.02f}".format(b_score))

    return scores


def get_all_info(img_paths):
    labels = []
    masks = []
    for path in img_paths:
        m, l = get_item_info(path)
        labels.append(l)
        masks.append(m)
    masks = np.array(masks)
    labels = np.array(labels)
    return masks, labels


def get_item_info(ref_path: str):
    item_info = info.loc[info["image_path"] == ref_path]

    mask_path = item_info["mask_path"].item()
    label = item_info["label"].item()

    mask = resize(image=np.array(Image.open("../" + mask_path)))
    return mask["image"], label


def get_patient_info(ref_path, img_paths):
    pid = info.loc[info["image_path"] == ref_path, "id"].item()
    patient_info = info.loc[info["id"] == pid]

    imgs = patient_info["image_path"].tolist()
    masks = patient_info["mask_path"].tolist()

    idx = [i for i, path in enumerate(img_paths) if path in imgs]
    return idx, imgs, masks
