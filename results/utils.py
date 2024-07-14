import torch
import torch.nn as nn
import torch.nn.functional as nnF

import os

import pandas as pd
import numpy as np
import torch

from matplotlib.axes import Axes
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A


class Dice(nn.Module):
    def __init__(self, multilabel=False, smooth=1e-6, dims=(-2, -1)):
        super(Dice, self).__init__()
        self.multilabel = multilabel
        self.smooth = smooth
        self.dims = dims

    def forward(self, output, target):
        output = output.sigmoid()
        output[output > 0.5] = 1

        if not self.multilabel:
            return self._dice(output, target).mean()

        target = nnF.one_hot(target, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()
        dice_per_class = torch.stack([self._dice(output[:, c], target[:, c]) for c in range(output.shape[1])], dim=1)
        return dice_per_class.mean(dim=1).mean()

    def _dice(self, output, target):
        tp = (output * target).sum(dim=self.dims)
        fp = (output * (1 - target)).sum(dim=self.dims)
        fn = ((1 - output) * target).sum(dim=self.dims)
        return (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)


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

    segmentation = y_pred.sigmoid().unsqueeze(2).repeat(1, 1, 3)
    segmentation[:, :, :2] = 0
    ax.imshow(segmentation, alpha=0.8)

    true_segmentation = y_true.unsqueeze(2).repeat(1, 1, 3)
    true_segmentation[:, :, 1:] = 0
    ax.imshow(true_segmentation, alpha=0.8)


class Results:
    def __init__(self, multilabel) -> None:
        self.multilabel = multilabel
        self.scoring = Dice(multilabel=multilabel)

    def show_patient_results(self, patient_info, title):
        img_idx, mean, std = patient_info
        n_imgs = len(img_idx)
        fig, axes = plt.subplots(1, n_imgs, sharey=True, figsize=(n_imgs * 3, 3))
        fig.suptitle(title.format(mean, std))

        if n_imgs == 1:
            axes = [axes]

        for ax, idx in zip(axes, img_idx):
            y_pred, y_true, img = self._get_item(idx, path=True)
            ax.set_title("{:02.02f}%".format(self.scoring(y_pred, y_true).item() * 100))
            ax.set_axis_off()
            show_image(y_true, y_pred, img, ax)

    def calculate_patient_scores(self):
        available_indices = set(range(len(self.fold["paths"])))
        p_scores = []

        while len(available_indices) != 0:

            pid_fold = available_indices.pop()
            p_img_ref = self.fold["paths"][pid_fold]
            p_idx = get_patient_info(p_img_ref, self.fold)

            # calcula os resultados de cada imagem do paciente
            f1 = np.array([self.scoring(*e) for i, e in enumerate(zip(self._get_item())) if i in p_idx])
            p_scores.append((p_idx, f1.mean(), f1.std()))

            available_indices -= set(p_idx)

        return p_scores

    def show_results(self, file: dict[str, dict]):
        for name, fold in file.items():
            self.fold = fold

            score = self.scoring(self._get_item()).item() * 100
            print("F1 macro: {:02.02f}%".format(score))

            p_scores = self.calculate_patient_scores()
            p_scores.sort(key=lambda x: x[1])

            worst_patient = p_scores[0]
            self.show_patient_results(worst_patient, name + " worst f1 macro score: {:02.02f} +- {:02.04f}")

            best_patient = p_scores[-1]
            self.show_patient_results(best_patient, name + " best f1 macro score: {:02.02f} +- {:02.04f}")

    def _get_item(self, id=None, path: bool = False):
        if id is not None:
            if path:
                return self.fold["y_pred"][id], self.fold["y_true"][id], self.fold["paths"][id]
            return self.fold["y_pred"][id], self.fold["y_true"][id]

        if path:
            return self.fold["y_pred"], self.fold["y_true"], self.fold["paths"]
        return self.fold["y_pred"], self.fold["y_true"]


label_to_channel = {"glioma": 0, "meningioma": 1, "pituitary": 2}


def _to_multichannel(x, label):
    channel = label_to_channel[label]
    x_ = torch.zeros_like(x).unsqueeze(2).repeat(1, 1, 3)
    x_[channel] = x
    return x_


info = pd.read_csv("../dataset/data/info.csv")


def get_patient_info(ref_path, fold, return_label=False):
    pid = info.loc[info["image_path"] == ref_path, "id"].item()
    imgs = info.loc[info["id"] == pid, "image_path"].tolist()

    if return_label:
        label = info.loc[info["id"] == pid, "label"][0].item()
        return [i for i, path in enumerate(fold["paths"]) if path in imgs], label

    return [i for i, path in enumerate(fold["paths"]) if path in imgs]
