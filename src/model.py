import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, model_arch, n_outputs, activation=None, pretrained=True):
        super(CNN, self).__init__()

        weights = 'DEFAULT' if pretrained else None

        if model_arch == "resnet101":
            self.model = models.resnet101(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, n_outputs)
        
        elif model_arch == "fcn_resnet101":
            self.model = models.segmentation.fcn_resnet101(weights=weights, weights_backbone=weights)
            in_channels = self.model.classifier[-1].in_channels
            self.model.classifier[-1] = nn.Conv2d(in_channels, n_outputs, kernel_size=1)
            self.model.aux_classifier = None

        else:
            raise NotImplementedError(
                f'The "{model_arch}" model is not implemented. Please choose one'
                ' of the following: "resnet101" or "fcn_resnet101".'
            )

        self.model_arch = model_arch
        self.activation = activation or nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self._model_out_parser(x)
        x = self.activation(x)
        return x

    def _model_out_parser(self, x):
        return x["out"] if self.model_arch == "fcn_resnet101" else x
