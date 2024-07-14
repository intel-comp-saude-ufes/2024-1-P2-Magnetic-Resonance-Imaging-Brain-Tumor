import torch
import torch.nn as nn
import torch.nn.functional as nnF

import os

import pandas as pd
import numpy as np
import torch

from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A


class Dice(nn.Module):
    def __init__(self, smooth=1e-6, dims=(-2, -1)):
        super(Dice, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, output, target):
        output = output.sigmoid()
        output[output > 0.5] = 1

        tp = (output * target).sum(dim=self.dims)
        fp = (output * (1 - target)).sum(dim=self.dims)
        fn = ((1 - output) * target).sum(dim=self.dims)
        return ((2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)).mean()


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


class Results:
    def __init__(self, files: dict[str, dict]) -> None:
        self.files = files
        self.scoring = Dice()
        self.scores = {}

    def show_patient_results(self, patient_info, title):
        img_idx, mean, std = patient_info
        n_imgs = len(img_idx)
        cols = min(n_imgs, 6)
        rows = (n_imgs // (cols + 1)) + 1
        fig, axes = plt.subplots(rows, cols, sharey=True, figsize=(cols * 3, 3.1 * rows))
        fig.suptitle(title.format(mean, std))

        if cols == 1:
            axes = np.array(axes)
        axes = axes.flatten()
        for ax in axes:
            ax.set_axis_off()

        for ax, idx in zip(axes, img_idx):
            y_pred, y_true, img = self._get_item(idx, path=True)
            ax.set_title("{:02.02f}%".format(self.scoring(y_pred, y_true).item() * 100))
            show_image(y_true, y_pred, img, ax)

    def calculate_patient_scores(self):
        n_indices = len(self.fold["paths"])
        available_indices = set(range(len(self.fold["paths"])))
        p_scores = []

        while len(available_indices) != 0:
            pid_fold = available_indices.pop()
            p_img_ref = self.fold["paths"][pid_fold]
            p_idx = get_patient_info(p_img_ref, self.fold)

            # calcula os resultados de cada imagem do paciente
            f1 = np.array([self.scoring(*self._get_item(i)) for i in range(n_indices) if i in p_idx])
            p_scores.append((p_idx, f1.mean(), f1.std()))

            available_indices -= set(p_idx)

        return p_scores

    def show(self):
        for name, fold in self.files.items():
            self.fold = fold

            score = self.scoring(*self._get_item()).item() * 100
            print(name, "F1 macro: {:02.02f}%".format(score))
            self.scores[name] = score

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

    def boxplot(self):
        scores = np.array(list(self.scores.values()))
        print(scores.mean(), scores.std())


info = pd.read_csv("../dataset/data/info.csv")


def get_patient_info(ref_path, fold, return_label=False):
    pid = info.loc[info["image_path"] == ref_path, "id"].item()
    imgs = info.loc[info["id"] == pid, "image_path"].tolist()

    if return_label:
        label = info.loc[info["id"] == pid, "label"][0].item()
        return [i for i, path in enumerate(fold["paths"]) if path in imgs], label

    return [i for i, path in enumerate(fold["paths"]) if path in imgs]
