import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(
        self, model_arch, n_outputs, activation=None, pretrained=True, freeze_conv=True
    ):
        super(CNN, self).__init__()

        if model_arch == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = models.densenet121(weights=weights)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, n_outputs)

        elif model_arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, n_outputs)

        elif model_arch == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            self.model = models.vgg16(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, n_outputs)

        else:
            raise NotImplementedError(
                f'The "{model_arch}" model is not implemented. Please choose one'
                ' of the following: "densenet121", "resnet18" or "vgg16".'
            )

        self.activation = activation or nn.Identity()

        if freeze_conv:
            self._freeze_convolutional(model_arch)

    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x

    def _freeze_convolutional(self, model_arch):
        for param in self.model.parameters():
            param.requires_grad = False

        parameters = None
        if model_arch in ["densenet121", "vgg16"]:
            parameters = self.model.classifier.parameters()
        elif model_arch == "resnet18":
            parameters = self.model.fc.parameters()
        else:
            raise NotImplementedError(
                f'The "{model_arch}" model is not implemented. Please choose one'
                ' of the following: "densenet121", "resnet18" or "vgg16".'
            )

        for param in parameters:
            param.requires_grad = True
