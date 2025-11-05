from typing import List
import torch
import numpy as np
import yaml
from tools.yaml_config import FullCertificationConfig
from networks import ReLUNN
from pydantic import ValidationError
import datetime

from bounds import (
    compute_bounds,
    check_stability_neurons,
    prune_adversarial_targets,
)
from tools import (
    add_functions_to_class,
    get_project_path,
    create_folder,
    round_list_depth_2,
    round_list_depth_3,
    change_to_zero_negative_values,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@add_functions_to_class(
    compute_bounds, check_stability_neurons, prune_adversarial_targets
)
class Solver:
    def __init__(
        self,
        network: ReLUNN,
        epsilon: float,
        x: List[float],
        ytrue: int,
        verbose: bool = False,
        L: List[List[float]] = None,
        U: List[List[float]] = None,
        use_inactive_neurons: bool = False,
        use_active_neurons: bool = False,
        **kwargs,
    ):
        self.network = network.to(device)
        self.K = network.K
        self.n = network.n
        self.W = round_list_depth_3(network.W)
        self.b = round_list_depth_2(network.b)

        self.n = np.array(self.n)
        self.W = [np.array(self.W[k - 1]) for k in range(1, self.K + 1)]

        self.b = [np.array(self.b[k]) for k in range(self.K)]


        self.dataset = kwargs.get("dataset")
        self.epsilon = epsilon
        self.x = x
        self.ytrue = ytrue

        self.ytarget = kwargs.get("ytarget", None)

        if "Lan" in self.__class__.__name__ and self.ytarget is not None:
            self.ytargets = [self.ytarget]
        else:
            self.ytargets = [
                j for j in range(self.n[self.K]) if j != self.ytrue
            ]

        self.bounds_method = kwargs.get("bounds_method")
        self.L = L
        self.U = U
        if self.L is None or self.U is None:
            self.compute_bounds(
                method=self.bounds_method,
            )

        print(
            "STUDY : use active neurons: ",
            use_active_neurons,
            "use inactive neurons: ",
            use_inactive_neurons,
        )

        self.L = [np.array(self.L[k]) for k in range(self.K + 1)]
        self.U = [np.array(self.U[k]) for k in range(self.K + 1)]

        self.use_inactive_neurons = use_inactive_neurons
        self.use_active_neurons = use_active_neurons
        self.check_stability_neurons(
            use_active_neurons=use_active_neurons,
            use_inactive_neurons=use_inactive_neurons,
        )
        # print('STUDY in generic solver: stable inactive neurons: ', self.stable_inactives_neurons)
        # print('STUDY in generic solver: stable active neurons: ', self.stable_actives_neurons)
        # for k in range(len(self.L)):
        #     for j in range(len(self.L[k])):
        #         self.L[k][j] -= 1
        #         self.U[k][j] += 1


        self.U_above_zero = change_to_zero_negative_values(
            self.U, dim=2
        )  # ATTENTION U N'EST PAS PRECIS : POUR EVITER CAS DES NEURONES STABLES INACTIFS
        self.L_above_zero = change_to_zero_negative_values(
            self.L, dim=2
        )  # ATTENTION : CECI POSERA UN PROBLEME POUR LES CONTRAINTES TRIANGULAIRES

        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.prune_adversarial_targets()

        # Initialize other parameters of the run
        self.is_robust = None
        self.best_adversarial_examples = None
        self.verbose = verbose

        self.name = kwargs.get("certification_model_name")

        self.folder_name = kwargs.get("folder_name", None)

        if self.folder_name is None:
            self.folder_name = "results"

        create_folder(f"{self.folder_name}/{self.name}")

        self.benchmark_dataframe = None
        self.data_index = kwargs.get("data_index", 0)
        self.network_name = kwargs.get("network_name", "ReLUNN")
        self.dataset_name = kwargs.get("dataset_name")

        self.is_trivially_solved = self.ytargets == []
        print("STUDY: is trivially solved: ", self.is_trivially_solved)

    @staticmethod
    def parse_yaml(yaml_file):
        with open(yaml_file, "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            validated_config = FullCertificationConfig(**raw_config)
        except ValidationError as e:
            print(f"Erreur de validation du fichier YAML :\n{e}")
            print("raw config:", raw_config)
            raise

        return dict(
            dataset=validated_config.data.name,
            network=ReLUNN.from_yaml(yaml_file),
            epsilon=validated_config.epsilon,
            x=validated_config.data.x,
            ytrue=validated_config.data.y,
            ytarget=validated_config.data.ytarget,
            bounds_method=validated_config.data.bounds_method,
            L=validated_config.data.L,
            U=validated_config.data.U,
        )

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        params = cls.parse_yaml(yaml_file)
        return cls(**params, **kwargs)
