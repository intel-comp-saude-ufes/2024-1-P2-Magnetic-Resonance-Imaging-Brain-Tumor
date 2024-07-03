from dotenv import load_dotenv
from os.path import exists
import argparse
import wandb


def parse_arguments():
    parse = argparse.ArgumentParser(description="Brain MRI prediction")
    parse.add_argument("--segmentation", help="Enable segmentation task", action=argparse.BooleanOptionalAction, default=False)
    parse.add_argument("--multilabel", help="Enable binary segmentation", action=argparse.BooleanOptionalAction, default=False)
    parse.add_argument("--max-epochs", type=int)
    parse.add_argument("--batch-size", type=int)

    args = parse.parse_args()
    return vars(args)


def initialize_wandb(config: dict):
    """Initialize wandb project run.

    Args:
        config (dict): run's configurations.
    """
    if not exists("./wandb.env"):
        print("O arquivo './wandb.env' não existe. A execução não será salva.")
        return

    load_dotenv(dotenv_path="./wandb.env")

    wandb.init(
        project="2024-1-P2-TIC",
        config=config,
    )
