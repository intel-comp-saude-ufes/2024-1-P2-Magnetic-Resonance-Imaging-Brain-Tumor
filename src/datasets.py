from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_train = A.Compose(
    [
        A.Resize(224, 224),
        A.ColorJitter(brightness=0.4, contrast=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # discutir importância
        ToTensorV2(transpose_mask=True),
    ]
)

transform_eval = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True),
    ]
)


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        segmentation=False,
        is_binary=False,
        transform=None,
    ):
        """
        o dataset precisa armazenar as seguintes informações (arquivo info.csv):
        ### id, image_path, mask_path, label ###

        o id é o id do paciente, vai ser usado pra dividir,
        treino/teste e se for fazer crossvalidation
        StratifiedGroupKFold

        seria interessante uma flag pra indicar se é classificação/segmentação
        e uma flag pra indicar se a segmentação é binária/multiclasse
        """
        self.segmentation = segmentation

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.data = np.array("dataset/" + metadata["image_path"])
        self.groups = np.array(metadata["id"])
        self.labels = np.array(
            metadata["label"].map(
                # {"no_tumor": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}
                {"glioma": 0, "meningioma": 1, "pituitary": 2}
            )
        )
        self.masks = np.array("dataset/" + metadata["mask_path"])

        if self.segmentation:
            self.is_binary = is_binary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        as transformações devem ser aplicadas na imagem e na mascara ao
        mesmo tempo (caso for segmentação), ja que uma depende da outra,
        dar uma olhada na biblioteca 'Albumentations' -> CITAR NO ARTIGO

        a mascara é um tensor [h, w] (não converter pra RGB). se for
        segmentação multiclasse precisa transformar a mascara em um tensor
        [c, h, w], onde apenas a posição respectiva a label é preenchida
        na dimensão c

        exemplo:

        c, h, w = 3, 2, 2

        label = 1

        mask = [[1, 0],
                [0, 1]]

        new_mask = [[[0, 0], [[1, 0], [[0, 0],
                    [0, 0]], [0, 1]], [0, 0]]]


        se for classificação retorna label como o label original mesmo (int)
        se for segmentação retorna a mascara no lugar de label
        """

        img_path = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        mask_path = self.masks[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        label = self.labels[idx]

        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]

        if self.segmentation:
            mask: torch.Tensor = transformed["mask"]

            if not self.is_binary:
                mask_ = torch.zeros((3, *mask.shape))
                mask_[label] = mask
                return img, mask_, img_path

            return img, mask, img_path

        return img, label, img_path


from sklearn.model_selection import StratifiedGroupKFold


def getDatasets(random_state):
    metadata = pd.read_csv("./dataset/data/info.csv")
    metadata = metadata[metadata["label"] != "no_tumor"]
    X = metadata["image_path"]
    labels = metadata["label"]
    groups = metadata["id"]

    gss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
    train, test = next(gss.split(X, labels, groups))

    print(
        "Is there an intersection of groups between train & test? ",
        set(metadata.iloc[train]["id"]).intersection(metadata.iloc[test]["id"]),
    )

    print(
        "Distribution of classes: \n\tTest:\n",
        metadata.iloc[train]["label"].value_counts(normalize=True).mul(100).round(2),
        "\n\tTrain:\n",
        metadata.iloc[test]["label"].value_counts(normalize=True).mul(100).round(2),
    )

    train_df = metadata.iloc[train]
    gss = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
    train_, val = next(
        gss.split(train_df["image_path"], train_df["label"], train_df["id"])
    )

    print(
        "Is there an intersection of groups between train & val? ",
        set(train_df.iloc[train_]["id"]).intersection(train_df.iloc[val]["id"]),
    )

    print(
        "Distribution of classes: \n\tTrain:\n",
        train_df.iloc[train_]["label"].value_counts(normalize=True).mul(100).round(2),
        "\n\tValidation:\n",
        train_df.iloc[val]["label"].value_counts(normalize=True).mul(100).round(2),
    )

    problem = dict(segmentation=True, is_binary=False)

    train_dataset = BrainTumorDataset(
        train_df.iloc[train_], transform=transform_train, **problem
    )
    test_dataset = BrainTumorDataset(
        metadata.iloc[test], transform=transform_eval, **problem
    )
    val_dataset = BrainTumorDataset(
        train_df.iloc[val], transform=transform_eval, **problem
    )

    return train_dataset, test_dataset, val_dataset
