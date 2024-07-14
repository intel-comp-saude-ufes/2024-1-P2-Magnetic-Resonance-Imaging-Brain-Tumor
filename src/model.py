from collections import OrderedDict
from torch.nn import functional as nnF
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, n_outputs, pretrained=True):
        super(CNN, self).__init__()

        weights = "DEFAULT" if pretrained else None

        model = models.segmentation.deeplabv3_resnet101(weights=weights, weights_backbone=weights)
        in_channels = model.classifier[-1].in_channels

        self.backbone = nn.Sequential(*list(model.backbone.children()))

        self.segmentation = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=n_outputs)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        features = self.backbone(x)

        classification = self.classification(features)

        segmentation = self.segmentation(features)
        segmentation = nnF.interpolate(segmentation, size=input_shape, mode="bilinear", align_corners=False)
        segmentation = segmentation.squeeze(1)

        return OrderedDict([
            ("seg", segmentation),
            ("class", classification)
        ])
