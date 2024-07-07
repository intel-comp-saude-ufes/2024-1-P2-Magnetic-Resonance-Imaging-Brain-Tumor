from PIL import Image
import h5py
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm


def load_data(file_path):
    try:
        with h5py.File(file_path, "r") as data:
            data = data["cjdata"]

            pid = tuple(map(int, data["PID"][:].flatten()))
            image = data["image"][:]
            mask = data["tumorMask"][:]
            label = int(data["label"][:].item())

            return pid, image, mask, label

    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None, None, None


def preprocess_pid(pids, count_pid, pid):
    if pid not in pids:
        pids[pid] = count_pid
        count_pid += 1
    return pids[pid], count_pid


def preprocess_and_save_image(image, image_save_path):
    image_normalized = ((image - image.min()) / (image.max() - image.min())) * 255
    image_uint8 = image_normalized.astype(np.uint8)
    image_pil = Image.fromarray(image_uint8, "L")
    image_pil.save(image_save_path)


def preprocess_and_save_mask(mask, mask_save_path):
    mask[mask != 0] = 255
    mask_pil = Image.fromarray(mask, "L")
    mask_pil.save(mask_save_path)


def preprocess_label(label):
    label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}

    return label_map.get(label, "unknown")


def process(path, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    pids = {}
    count_pid = 0

    info = []
    count_total = 0

    with tqdm(os.listdir(path), desc="[ FIGSHARE ]") as listdir:
        for file in listdir:
            file_path = os.path.join(path, file)
            pid, image, mask, label = load_data(file_path)

            if pid is not None:

                pid, count_pid = preprocess_pid(pids, count_pid, pid)

                image_save_path = os.path.join(output_dir, "images", f"image_{count_total:04d}.png")
                preprocess_and_save_image(image, image_save_path)

                mask_save_path = os.path.join(output_dir, "masks", f"mask_{count_total:04d}.png")
                preprocess_and_save_mask(mask, mask_save_path)

                label = preprocess_label(label)

                info.append((pid, "dataset/" + image_save_path, "dataset/" + mask_save_path, label))
                count_total += 1

    return info


def save_metadata(info, filename):
    df = pd.DataFrame(info, columns=["id", "image_path", "mask_path", "label"])
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_figshare.py <path_to_dir>")
        exit()

    path = sys.argv[1]
    output_dir = "data"
    metadata_filename = os.path.join(output_dir, "info.csv")

    info = process(path, output_dir)
    save_metadata(info, metadata_filename)
