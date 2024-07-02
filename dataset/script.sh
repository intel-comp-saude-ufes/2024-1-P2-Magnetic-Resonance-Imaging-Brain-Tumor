#!/bin/bash

FIGSHARE_URL="https://figshare.com/ndownloader/articles/1512427/versions/5"
FIGSHARE_ZIP="figshare_dataset.zip"
FIGSHARE_DATASET_FOLDERS=(
    brainTumorDataPublic_1-766
    brainTumorDataPublic_767-1532
    brainTumorDataPublic_1533-2298
    brainTumorDataPublic_2299-3064
)
KAGGLE_DATASET="ahmedhamada0/brain-tumor-detection"
KAGGLE_ZIP="brain-tumor-detection.zip"
TEMP_FOLDER="tmp"

check_error() {
    if [ $? -ne 0 ]; then
        echo "Error encountered. Exiting."
        exit 1
    fi
}

mkdir -p "$TEMP_FOLDER"
check_error

wget -O "$FIGSHARE_ZIP" "$FIGSHARE_URL"
check_error

for folder in "${FIGSHARE_DATASET_FOLDERS[@]}"; do
    unzip -q "$FIGSHARE_ZIP" "$folder.zip"
    check_error
    unzip -q "$folder.zip" -d "$TEMP_FOLDER"
    check_error
    rm "$folder.zip"
done

python3 process_figshare.py "$TEMP_FOLDER"
check_error

rm -r "${TEMP_FOLDER:?}/"*
check_error

# TO USE:
# https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#installation
# https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials
kaggle datasets download -d "$KAGGLE_DATASET"
check_error

unzip -q "$KAGGLE_ZIP" "no/*" -d "$TEMP_FOLDER"
check_error

python3 process_br35.py "${TEMP_FOLDER}/no"
check_error

rm -rf "$TEMP_FOLDER"
check_error

echo "All tasks completed successfully."
