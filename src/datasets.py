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


class CrossValidation:
    def __init__(
        self,
        metadata: pd.DataFrame,
        folds: list[np.ndarray],
        val_size: int = 4,
        random_state: int = 10,
        **kwargs,
    ):
        self.metadata = metadata
        self.folds = folds
        self.current_fold = 0
        self.val_size = val_size
        self.random_state = random_state
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
        train_val, test = self.folds[fold_name]
        self.current_fold += 1

        train, val = _split_dataset(self.metadata.iloc[train_val], size=self.val_size, random_state=self.random_state)

        return (
            fold_name,
            (
                BrainTumorDataset(metadata=self.metadata.iloc[train], **self.kwargs),
                BrainTumorDataset(metadata=self.metadata.iloc[test], **self.kwargs),
                BrainTumorDataset(metadata=self.metadata.iloc[val], **self.kwargs),
            ),
        )


from sklearn.model_selection import StratifiedGroupKFold


def _split_dataset(dataset: pd.DataFrame, size: int, return_all_folds=False, random_state: int = 10) -> tuple[np.ndarray, np.ndarray] | dict[str : tuple[np.ndarray, np.ndarray]]:
    gss = StratifiedGroupKFold(n_splits=size, shuffle=True, random_state=random_state)

    folds = {}
    for i, f in enumerate(gss.split(dataset["image_path"], dataset["label"], dataset["id"])):
        if not return_all_folds:
            return f
        fold_str = "fold_{:02d}".format(i)
        folds[fold_str] = f

    return folds


def split_datasets(
    dataset: pd.DataFrame,
    random_state: int = 10,
    segmentation=False,
    n_classes: int = 4,
    validation_size: int = 4,
    test_size: int = 10,
    cv: int = None,
):
    problem = dict(segmentation=segmentation, n_classes=n_classes)

    if cv:
        folds = _split_dataset(dataset, size=cv, return_all_folds=True, random_state=random_state)
        return CrossValidation(dataset, folds, validation_size, transform=transform_eval, **problem)

    train_val, test = _split_dataset(dataset, size=test_size, random_state=random_state)
    test_dataset = BrainTumorDataset(dataset.iloc[test], transform=transform_eval, **problem)

    train, val = _split_dataset(dataset.iloc[train_val], size=validation_size, random_state=random_state)
    train_dataset = BrainTumorDataset(dataset.iloc[train], transform=transform_train, **problem)
    val_dataset = BrainTumorDataset(dataset.iloc[val], transform=transform_eval, **problem)

    return [(None, (train_dataset, test_dataset, val_dataset))]
