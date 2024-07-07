import argparse


def parse_arguments():
    parse = argparse.ArgumentParser(description="Brain MRI prediction")
    parse.add_argument("--multilabel", help="Enable label segmentation", action=argparse.BooleanOptionalAction, default=False)
    parse.add_argument("--max-epochs", type=int)
    parse.add_argument("--batch-size", type=int)
    parse.add_argument("--cv", type=int, default=None)
    parse.add_argument("--val-size", help="Number of splits to execute, therefore, 1/n validation size", type=int, default=4)
    parse.add_argument("--test-size", help="Number of splits to execute, therefore, 1/n test size", type=int, default=10)
    parse.add_argument("--tensor-board", action=argparse.BooleanOptionalAction, default=False)

    args = parse.parse_args()
    return vars(args)
