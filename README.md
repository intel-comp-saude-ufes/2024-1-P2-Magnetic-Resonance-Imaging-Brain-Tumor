# 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor-Classification

## Installation

### Access your machine with GPU support and clone the repository
```
# SSH to Linux server via designated port (for tensorboard)
ssh -L 6006:LocalHost:6006 username@server_address

# Clone the repository
git clone git@github.com:intel-comp-saude-ufes/2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor.git
```

## We recommend setting a python virtual environment to install the required packages
```
cd 2024-1-P2-Magnetic-Resonance-Imaging-Brain-Tumor
python -m venv myenv 
source myvenv/bin/activate
pip install -r requirements.txt
```
## If you would like to use the same dataset as we did, run the following script 
**Note: you may need a Kaggle account**
```
cd datasets
bash script.sh
```
## Now you can run our program based on your needs
**Note: To use tensor board you must open the application via the ssh port on the preferred browser, example address: http://localhost:6006/**

-Options:
* --segmentation: Enable segmentation task (default: False);
* --multilabel: Enable binary segmentation (default: False);
* --max-epochs: Maximum number of epochs ;
* --batch-size: Batch size;
* --cv: Number of cross validation folders (default: None);
* --tensor-board: Use tensor board to track loss logs
```
cd ..
python3 ./main.py # **flags 
```


