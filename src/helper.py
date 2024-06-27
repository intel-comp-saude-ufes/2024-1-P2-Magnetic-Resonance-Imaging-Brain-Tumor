import argparse

def parse_arguments():
    parse = argparse.ArgumentParser(description='Brain MRI prediction')
    parse.add_argument('-tr', '--training_path', help='Path to training dataset')
    parse.add_argument('-te', '--test_path', help='Path to test dataset')
    parse.add_argument('-m', '--model', help='Model (densenet121 / resnet18 / vgg16)')
    parse.add_argument('-e', '--epochs', help='Epochs', type=int)
    parse.add_argument('-b', '--batchsize', help='Batch Size', type=int)
    args = parse.parse_args()
    return args