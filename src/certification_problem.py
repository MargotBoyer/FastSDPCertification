import numpy as np
import yaml
import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from solve.generic_solver import Solver
from solve import LayersValues
import solve
from tools import FullCertificationConfig
from pydantic import BaseModel
import pandas as pd
import datetime
import shutil
import argparse
import multiprocessing as mp

from tools import create_folder_benchmark
from solve.mosek_solve import concat_dataframes_with_missing_columns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networks
import data

from tools import get_project_path

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Certification_Problem:
    def __init__(
        self, network: networks.ReLUNN, epsilon: float, dataset: TensorDataset, **kwargs
    ):
        """Initialize the certification problem.
        Args:
            network (models.ReLUNN): The neural network model.
            epsilon (float): The perturbation bound.
            dataset (TensorDataset): The dataset for certification.
        """
        print("Initializing Certification Problem ...")
        self.network = network.to(device_ if kwargs.get("use_cuda", True) else "cpu")
        self.epsilon = epsilon
        self.dataset = dataset
        self.models = kwargs.get("models", [])
        self.network_name = kwargs.get("network_name", network.name)
        self.dataset_name = kwargs.get("dataset_name")
        self.yaml_file = kwargs.get("yaml_file", None)
        print("Data name in certification problem:", self.dataset_name)

        print("dataset in certification problem:", self.dataset)

        self.title = f"{self.network_name}-{self.epsilon}"
        if not os.path.exists(get_project_path(f"results/benchmark/{self.title}")):
            os.makedirs(get_project_path(f"results/benchmark/{self.title}"))

        print("Certification Problem initialized !")

    @classmethod
    def load_from_yaml(cls, yaml_file):
        """
        Load the certification problem from a YAML file.
        Args:
            yaml_file (str): Path to the YAML file.
        Returns:
            Certification_Problem: An instance of the Certification_Problem class.
        """
        print(f"Loading certification problem from {yaml_file} ...")
        network = networks.ReLUNN.from_yaml(f"config/{yaml_file}")
        if network is not None:
            print("Network loaded successfully.")
        else:
            print("Failed to load network.")
            return None
        print("Loading dataset ...")
        dataset = data.load_dataset(f"config/{yaml_file}")
        if dataset is not None:
            print("Dataset loaded successfully.")
        else:
            print("Failed to load dataset.")
            return None
        print("Loading epsilon ...")
        with open(f"config/{yaml_file}", "r") as file:
            config = yaml.safe_load(file)
            print("CONFIG  inf CERTIFICATION PROBLEM:     ", config)
            epsilon = config["epsilon"]
        validated_config = FullCertificationConfig(**config)
        print("Data name from config:", validated_config.data.name)
        return cls(
            network,
            epsilon,
            dataset,
            models=validated_config.models,
            network_name=validated_config.network.name,
            dataset_name=validated_config.data.name,
            yaml_file=yaml_file,
        )

    def __str__(self):
        """
        String representation of the certification problem.
        """
        return f"Certification Problem with epsilon: {self.epsilon}, dataset size: {len(self.dataset)}"

    def run(self, solver_config: BaseModel, title_run: str = "") -> None:
        """
        Run the certification problem.
        """
        model_class = getattr(solve, solver_config.certification_model_name)
        print(
            f"Running certification with solver: {solver_config.certification_model_name}"
        )

        print("SOLVER CONFIG:", solver_config)
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        stable_actives_study = pd.DataFrame(
            columns=[
                "label",
                "data_index",
                "Number_actives_stable",
                "Number_inactives_stable",
                "Number_targets",
            ]
        )

        for i, (x, ytrue) in enumerate(dataloader):

            # print("x  :", x)
            print("ytrue:", ytrue)
            # if (ytrue.item()) % 10 != 0:
            #     print(
            #         f"Skipping sample {i + 1} with label {ytrue.item()} as it is not a multiple of 10."
            #     )
            #     continue
            # assert ytrue == y, "ytrue should match the label y"
            print("x  shape:", x.shape)
            if i <= 40 or i >= 45:
                # print(
                #     f"Stopping after 25 samples. Current sample index: {i}. You can change this limit in the code."
                # )
                print("Skipping data sample ", i + 1, "for testing purposes.")
                continue
            print("i : ", i)
            # exit()
            x = x.view(-1)  # Ensure x is a 2D tensor
            print("x  shape after view:", x.shape)
            print(
                f"STUDY : Running certification for sample {i + 1} of label {ytrue.item()}"
            )
            dict_infos = dict(solver_config)
            dict_infos.pop("certification_model_name")
            print("dict_infos:", dict_infos)

            # print("Network device : ", self.network.device)
            print("x device : ", x.device)
            x = x.to(device_)
            print("x device : ", x.device)
            y_pred = self.network(x)
            print("y_pred:", y_pred)

            model_instance = model_class(
                network=self.network,
                epsilon=self.epsilon,
                x=x,
                ytrue=ytrue.item(),
                data_index=i,
                dataset_name=self.dataset_name,
                network_name=self.network_name,
                folder_name=f"results/benchmark/{self.title}/{title_run}",
                **dict_infos,
            )

            nb_actives = len(model_instance.stable_actives_neurons)
            nb_inactives = len(model_instance.stable_inactives_neurons)
            nb_targets = len(model_instance.ytargets)
            model_instance.solve(verbose=True)

            self.benchmark = concat_dataframes_with_missing_columns(
                self.benchmark, model_instance.benchmark_dataframe
            )
            self.benchmark.to_csv(
                f"results/benchmark/{self.title}/{title_run}/results.csv",
                index=False,
            )
            stable_actives_study = pd.concat(
                [
                    stable_actives_study,
                    pd.DataFrame(
                        {
                            "label": [ytrue],
                            "data_index": [i],
                            "Number_actives_stable": [nb_actives],
                            "Number_inactives_stable": [nb_inactives],
                            "Number_targets": [nb_targets],
                        }
                    ),
                ],
                ignore_index=True,
            )

        stable_actives_study.to_csv(
            f"results/benchmark/{self.title}/{title_run}/stable_actives_study.csv"
        )

    def solve(self, title_run: str = "") -> None:
        print("Starting certification problem solving ...")
        print("self.models:", self.models)

        title_run = (
            datetime.datetime.now().strftime("%m_%d_%Hh%M_%Ss") + "_" + title_run
        )

        self.benchmark = pd.DataFrame()

        if not os.path.exists(
            get_project_path(f"results/benchmark/{self.title}/{title_run}")
        ):
            os.makedirs(
                get_project_path(f"results/benchmark/{self.title}/{title_run}"),
                exist_ok=True,
            )

        if self.yaml_file is not None:
            shutil.copyfile(
                get_project_path(f"config/{self.yaml_file}"),
                get_project_path(
                    f"results/benchmark/{self.title}/{title_run}/{self.yaml_file}"
                ),
            )

        for i, model_config in enumerate(self.models):

            print("Solving with model:", model_config.certification_model_name)
            print("model dict :", model_config)
            self.run(model_config, title_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Network Parser")

    parser.add_argument("network", type=str, help="Network to test", default="6x100")
    parser.add_argument("title_run", type=str, help="Description-run", default="")
    args = parser.parse_args()

    print("Number of CPU : ", mp.cpu_count())

    yaml_file = f"{args.network}.yaml"  # "mnist_one_data_benchmark.yaml"
    certif_problem = Certification_Problem.load_from_yaml(yaml_file)
    certif_problem.solve(args.title_run)
