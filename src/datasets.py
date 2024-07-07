import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


transform_train = A.Compose(
    [
        A.Resize(224, 224),
        A.Affine(translate_percent=0.04, scale=(0.8, 1.2)),
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
        n_classes: int,
        transform=None,
    ):
        self.n_classes = n_classes
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.data = np.array(metadata["image_path"])
        self.groups = np.array(metadata["id"])
        self.labels = torch.Tensor(np.array(metadata["label"])).type(torch.int64)
        self.masks = np.array(metadata["mask_path"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = np.array(Image.open(img_path).convert("RGB"))

        mask_path = self.masks[idx]
        mask = np.array(Image.open(mask_path))

        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]

        mask: torch.Tensor = transformed["mask"]
        mask[mask != 0] = self.labels[idx]
        mask = mask.float() if self.n_classes == 1 else mask.long()

        return img, mask, img_path


class CrossValidation:
    def __init__(
        self,
        metadata: pd.DataFrame,
        folds: list[np.ndarray],
        n_classes: int,
        val_size: int,
        random_state: int = 10,
    ):
        self.metadata = metadata
        self.folds = folds
        self.current_fold = 0
        self.n_classes = n_classes

        self.val_size = val_size

        self.random_state = random_state

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

        _train_val = _split_dataset(
            self.metadata.iloc[train_val],
            self.val_size,
            random_state=self.random_state,
        )
        train, val = _train_val["fold_00"]

        return (
            fold_name,
            (
                BrainTumorDataset(metadata=self.metadata.iloc[train], n_classes=self.n_classes, transform=transform_train),
                BrainTumorDataset(metadata=self.metadata.iloc[test], n_classes=self.n_classes, transform=transform_eval),
                BrainTumorDataset(metadata=self.metadata.iloc[val], n_classes=self.n_classes, transform=transform_eval),
            ),
        )


from sklearn.model_selection import StratifiedGroupKFold


def _split_dataset(
    dataset: pd.DataFrame,
    splits: int,
    return_all_folds=False,
    random_state: int = 10,
) -> tuple[np.ndarray, np.ndarray] | dict[str : tuple[np.ndarray, np.ndarray]]:
    gss = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)

    folds = {}
    for i, f in enumerate(gss.split(dataset["image_path"], dataset["label"], dataset["id"])):
        fold_str = "fold_{:02d}".format(i)
        folds[fold_str] = f
        if not return_all_folds:
            break

    return folds


def split_datasets(
    dataset: pd.DataFrame,
    n_classes: int,
    test_size: int,
    val_size: int,
    cv: bool,
    random_state: int = 10,
):
    folds = _split_dataset(
        dataset,
        test_size,
        return_all_folds=cv,
        random_state=random_state,
    )

    return CrossValidation(
        dataset,
        folds,
        n_classes=n_classes,
        val_size=val_size,
        random_state=random_state,
    )


def prepare_dataloader(splits, batch_size):
    training, test, validation = splits
    train_loader = DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader
