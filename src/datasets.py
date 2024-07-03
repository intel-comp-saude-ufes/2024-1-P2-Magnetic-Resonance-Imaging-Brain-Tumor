import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms


transform_train = A.Compose(
    [
        A.Resize(224, 224),
        A.ColorJitter(brightness=0.4, contrast=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # discutir importância
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
        n_classes: int = 4,
        transform=None,
    ):
        self.segmentation = segmentation
        self.n_classes = n_classes

        self.transform = transform if transform is not None else transforms.ToTensor()

        self.data = np.array(metadata["image_path"])
        self.groups = np.array(metadata["id"])
        self.labels = torch.Tensor(np.array(metadata["label"])).type(torch.int64)
        self.masks = np.array(metadata["mask_path"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # TODO: fix opencv problem or change image reading/transformation pipeline
        img_path = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_path = self.masks[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        label = self.labels[idx]

        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]

        if self.segmentation:
            mask: torch.Tensor = transformed["mask"]
            if self.n_classes == 2:
                return img, mask, img_path

            mask_ = torch.zeros((self.n_classes, *mask.shape))
            mask_[label] = mask
            return img, mask_, img_path

        return img, label, img_path


class TestFolds:
    """Basicamente, essa função guarda os índices de cada fold de teste e as configurações dos datasets do experimento (segmentation, n_classes).
    Com essa classe, é possível fazer a divisão de folds dinamicamente e no final, testar cada uma separadamente, gerando um boxplot.
    """

    def __init__(self, metadata: pd.DataFrame, folds: list[np.ndarray], **kwargs):
        self.metadata = metadata
        self.folds = folds
        self.current_fold = 0
        self.kwargs = kwargs

    def __len__(self):
        fold_name = "fold_{:02d}".format(self.current_fold)
        return len(self.folds[fold_name])

    def __get_n_splits__(self):
        return len(self.folds)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fold >= len(self.folds):
            raise StopIteration

        fold_name = "fold_{:02d}".format(self.current_fold)
        idx = self.folds[fold_name]
        self.current_fold += 1

        return fold_name, BrainTumorDataset(metadata=self.metadata.iloc[idx], **self.kwargs)


from sklearn.model_selection import StratifiedGroupKFold


def get_dataset_splits(dataset: pd.DataFrame, size: int, random_state: int = 10, debug=False) -> tuple[np.ndarray, np.ndarray]:
    """Splits dataset.

    Args:
        dataset (pd.DataFrame): Dataset.
        size (int): Number of splits (inverse of size).
        random_state (int): Defaults to 10.
        debug (bool, optional): Prints each fold's distribution. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Train and test indices.
    """
    gss = StratifiedGroupKFold(n_splits=size, shuffle=True, random_state=random_state)
    train, test = next(gss.split(dataset["image_path"], dataset["label"], dataset["id"]))

    if debug:
        print("Train: ", dataset.iloc[train]["label"].value_counts(normalize=True).mul(100).round(2))
        print("Test: ", dataset.iloc[train]["label"].value_counts(normalize=True).mul(100).round(2))

    return train, test


def get_test_folds(dataset: pd.DataFrame, n_folds: int = 5, random_state: int = 10, debug=False):
    gss = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = {"fold_{:02d}".format(i): test for i, (_, test) in enumerate(gss.split(dataset["image_path"], dataset["label"], dataset["id"]))}

    if debug:
        for fold, idx in folds.items():
            print("Fold {:02d}".format(fold), dataset.iloc[idx]["label"].value_counts(normalize=True).mul(100).round(2))

    return folds


def split_datasets(
    dataset: pd.DataFrame,
    random_state: int = 10,
    segmentation=False,
    n_classes: int = 4,
    validation_size: int = 4,
    test_size: int = 10,
    n_test_folds: int = 5,
):
    problem = dict(segmentation=segmentation, n_classes=n_classes)

    train, test = get_dataset_splits(dataset, test_size, random_state)
    # test_dataset = BrainTumorDataset(dataset.iloc[test], transform=transform_eval, **problem)

    test_folds = get_test_folds(dataset.iloc[test], n_folds=n_test_folds, random_state=10)
    test_dataset = TestFolds(dataset.iloc[test], test_folds, transform=transform_eval, **problem)

    train_, val = get_dataset_splits(dataset.iloc[train], validation_size, random_state)
    train_dataset = BrainTumorDataset(dataset.iloc[train_], transform=transform_train, **problem)
    val_dataset = BrainTumorDataset(dataset.iloc[val], transform=transform_eval, **problem)

    return train_dataset, test_dataset, val_dataset
