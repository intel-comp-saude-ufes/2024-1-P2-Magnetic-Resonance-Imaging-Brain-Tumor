import pandas as pd
import os
from cv2 import resize, imread, IMREAD_COLOR, imwrite

CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]
COLORS = ["blue", "yellow", "green", "red"]
img_dir = "../archive"
resized_dir = "../resized"


def setup():
    if not os.path.exists(resized_dir):
        os.mkdir(resized_dir)


def index_images(file_dir: str, save_dir: str, size=(224, 224)):
    df = pd.DataFrame()
    image_paths = []
    labels = []

    for label, cl in enumerate(CLASSES):
        save_class_dir = os.path.join(save_dir, cl)
        img_class_dir = os.path.join(file_dir, cl)

        for item in os.listdir(img_class_dir):
            item_path = os.path.join(img_class_dir, item)
            img = imread(item_path, IMREAD_COLOR)

            resized_img = resize(img, size)

            resized_path = os.path.join(save_class_dir, item)
            image_paths.append(resized_path.removeprefix("../"))
            labels.append(label)
            imwrite(resized_path, resized_img)

    df["Brain_Image"] = image_paths
    df["Tumor"] = labels

    return df


def process_images(i):
    root = os.path.join(img_dir, i)
    save_dir = os.path.join(resized_dir, i)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

        for cl in CLASSES:
            class_dir = os.path.join(save_dir, cl)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

    print(root)
    df = index_images(root, save_dir)
    if not os.path.exists("../processed_data"):
        os.mkdir("../processed_data")

    df.to_csv(f"../processed_data/{i.lower()}_data.csv", index=False)
    return df


def pie(df, ax, title=""):
    ax.pie(
        df["Tumor"].value_counts(),
        labels=CLASSES,
        colors=COLORS,
        autopct="%.1f%%",
        explode=(0.025, 0.025, 0.025, 0.025),
        startangle=30,
    )
    ax.set_title(title)
