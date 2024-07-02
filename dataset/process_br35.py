from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import sys

def load_metadata(metadata_filename):
    df = pd.read_csv(metadata_filename)
    count_pid = max(df['id']) + 1
    return df, count_pid

def load_image(path):
    image = Image.open(path).convert('RGB')
    return image

def preprocess_and_save_image(image, image_save_path):
    image.save(image_save_path)

def preprocess_and_save_mask(image, mask_save_path):
    mask = np.zeros_like(np.asarray(image))
    mask_image = Image.fromarray(mask[:, :, 0], 'L')
    mask_image.save(mask_save_path)

def process(path, output_dir, metadata_filename):
    df, count_pid = load_metadata(metadata_filename)
    count_total = len(df)

    info = []
    with tqdm(os.listdir(path), desc='[ BR35 ]') as listdir:
        for i, file in enumerate(listdir):
            pid = count_pid + i

            image_path = os.path.join(path, file)
            image = load_image(image_path)

            image_save_path = os.path.join(output_dir, 'images', f'image_{count_total:04d}.png')
            preprocess_and_save_image(image, image_save_path)

            mask_save_path = os.path.join(output_dir, 'masks', f'mask_{count_total:04d}.png')
            preprocess_and_save_mask(image, mask_save_path)

            label = 'No tumor'

            info.append((pid, image_save_path, mask_save_path, label))
            count_total += 1

    df = pd.concat([df, pd.DataFrame(info, columns=df.columns)], ignore_index=True)
    df.to_csv(metadata_filename, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_br35.py <path_to_dir>")
        exit()

    path = sys.argv[1]
    output_dir = 'data'
    metadata_filename = os.path.join(output_dir, 'info.csv')

    process(path, output_dir, metadata_filename)
