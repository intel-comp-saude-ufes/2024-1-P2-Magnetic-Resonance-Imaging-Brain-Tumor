import torch
import torch.nn as nn
import torch.nn.functional as nnF


class Dice(nn.Module):
    def __init__(self, apply_sigmoid=True, multilabel=False, smooth=1e-6, dims=(-2, -1)):
        super(Dice, self).__init__()

        self.apply_sigmoid = apply_sigmoid
        self.multilabel = multilabel
        self.smooth = smooth
        self.dims = dims

    def forward(self, output, target):
        if self.apply_sigmoid:
            output = output.sigmoid()

        tp = (output * target).sum(dim=self.dims)
        fp = (output * (1 - target)).sum(dim=self.dims)
        fn = ((1 - output) * target).sum(dim=self.dims)
        return (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_sigmoid=True, multilabel=False, smooth=1e-6, dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()

        self.dice = Dice(apply_sigmoid, multilabel, smooth, dims)

    def forward(self, output, target):
        return 1 - self.dice(output, target)


class ComposedLoss(nn.Module):
    def __init__(self, loss_funcs, weights=None):
        super(ComposedLoss, self).__init__()

        self.loss_funcs = loss_funcs
        self.weights = ([1.0] * len(loss_funcs)) if weights is None else weights

    def forward(self, output, target):
        total_loss = 0.0
        for loss_func, weight in zip(self.loss_funcs, self.weights):
            total_loss += weight * loss_func(output, target)
        return total_loss
