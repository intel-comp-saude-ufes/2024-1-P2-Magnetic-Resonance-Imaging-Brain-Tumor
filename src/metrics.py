import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, apply_sigmoid=True, smooth=1e-6, dims=(-2, -1)):
        super(Dice, self).__init__()

        self.apply_sigmoid = apply_sigmoid
        self.smooth = smooth
        self.dims = dims

    def forward(self, output, target):
        if self.apply_sigmoid:
            output = output.sigmoid()

        tp = (output * target).sum(dim=self.dims)
        fp = (output * (1 - target)).sum(dim=self.dims)
        fn = ((1 - output) * target).sum(dim=self.dims)

        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, apply_sigmoid=True, smooth=1e-6, dims=(-2, -1)):
        super(DiceBCELoss, self).__init__()

        self.dice = Dice(apply_sigmoid, smooth, dims)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return (1 - self.dice(output, target)) + self.bce_loss(output, target)
