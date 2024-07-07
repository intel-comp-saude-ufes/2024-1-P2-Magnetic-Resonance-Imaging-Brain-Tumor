import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, n_outputs, activation=None, pretrained=True):
        super(CNN, self).__init__()

        weights = "DEFAULT" if pretrained else None

        self.model = models.segmentation.fcn_resnet101(weights=weights, weights_backbone=weights)
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_channels, n_outputs, kernel_size=1)
        self.model.aux_classifier = None

        self.n_outputs = n_outputs
        self.activation = activation or nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self._model_out_parser(x)
        x = self.activation(x)
        return x

    def _model_out_parser(self, x):
        x = x["out"]
        if self.n_outputs == 1:
            x = x.squeeze(1)
        return x
