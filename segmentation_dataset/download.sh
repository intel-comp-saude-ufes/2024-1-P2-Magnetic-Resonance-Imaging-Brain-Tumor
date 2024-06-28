#!/bin/bash

url="https://figshare.com/ndownloader/articles/1512427/versions/5"
output_zip="dataset.zip"
raw_dataset_folders=(
    brainTumorDataPublic_1-766
    brainTumorDataPublic_767-1532
    brainTumorDataPublic_1533-2298
    brainTumorDataPublic_2299-3064
)
main_dataset_folder="raw_dataset"

wget -O "$output_zip" "$url"

mkdir -p "$main_dataset_folder"

for folder in "${raw_dataset_folders[@]}"; do
    unzip -q "$output_zip" "$folder".zip
    unzip -q "$folder".zip -d "$main_dataset_folder"
    rm "$folder".zip
done

python process.py "$main_dataset_folder"
