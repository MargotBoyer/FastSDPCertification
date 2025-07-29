from adv_train import evaluate_robust, load_adversarial_training_config
import data
from network import ReLUNN
import argparse
from torch.utils.data import DataLoader
from tools import get_project_path
import torch
from train import evaluate

device = "cuda:0"

if __name__== "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Networks")

    parser.add_argument("network", type=str, help="Network to test", default="6x100")
    args = parser.parse_args()

    yaml_file = f"mnist-{args.network}.yaml"  # "mnist_one_data_benchmark.yaml"

    network = ReLUNN.from_yaml(f"config/{yaml_file}")
    network = network.to(device)


    config = load_adversarial_training_config(f"config/networks/{args.network}.yaml")
    robust_to_test_dataset = torch.load(
        get_project_path(config["evaluate_robustness_path"])
    )["dataset"]
    test_dataset = torch.load(get_project_path(config["test_path"]))  # ["dataset"]

    dataloader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,  # Pas besoin de mélanger pour test
        num_workers=2,
        pin_memory=True,
    )

    # pgd_robust = evaluate_robust(
    #                 network,
    #                 dataloader,
    #                 "cuda:0",
    #                 {
    #                     "eps": 0.015,
    #                     "alpha": 0.01,
    #                     "steps": 40,
    #                     "random_start": True,
    #                     "norm": "inf",
    #                 },
    #             )

    acc = evaluate(
                    network,
                    dataloader,
                )
    print("Accuracy: ", acc)