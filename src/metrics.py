import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, multilabel=False, smooth=1e-6, dims=(-2, -1)) -> None:
        super(Dice, self).__init__()

        self.multilabel = multilabel
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        if not self.multilabel:
            return self._dice(x, y).mean()

        dice_per_class = torch.stack([
            self._dice(x[:, c], y[:, c]) for c in range(x.shape[1])
        ], dim=1)
        return dice_per_class.mean(dim=1).mean()

    def _dice(self, x, y):
        tp = (x * y).sum(dim=self.dims)
        fp = (x * (1 - y)).sum(dim=self.dims)
        fn = ((1 - x) * y).sum(dim=self.dims)
        return (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, multilabel=False, smooth=1e-6, dims=(-2, -1)) -> None:
        super(SoftDiceLoss, self).__init__()

        self.dice = Dice(multilabel, smooth, dims)

    def forward(self, x, y):
        return 1 - self.dice(x, y)
