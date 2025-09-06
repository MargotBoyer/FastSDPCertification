import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time

from tools import round_list_depth_2, change_to_zero_negative_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_bounds_data(network, x, epsilon, n, K, method: str = "IBP"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    print(f"Computing bounds with method: {method} ...")
    print("epsilon : ", epsilon)
    L = [[-np.inf] * n[k] for k in range(K + 1)]
    U = [[np.inf] * n[k] for k in range(K + 1)]

    if method == "GREAT_BOUNDS":
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return

    norm = np.inf
    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)
    print("x shape : ", x.shape)
    
    bounded_model = BoundedModule(
        network,
        torch.zeros_like(x).to(device),
        bound_opts={"conv_mode": "patches"},
    )
    bounded_model.eval()

    ptb = PerturbationLpNorm(norm=norm, eps=epsilon)
    bounded_image = BoundedTensor(x, ptb)
    if method == "alpha-CROWN":
        lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    else:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    intermediate_bounds = bounded_model.save_intermediate()

    intermediate_bounds_list = list(intermediate_bounds.keys())

    # for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():
    #     print(f"{layer_name}:")
    #     if self.data_modele == "blob":
    #         print(f"  Min: {min_tensor.squeeze().cpu().numpy()}")
    #         print(f"  Max: {max_tensor.squeeze().cpu().numpy()}")
    #     else:
    #         print(f"  Min SHAPE: {min_tensor.squeeze().cpu().numpy().shape}")
    #         print("Min min : ", min_tensor.min())
    #         print("Min max : ", min_tensor.max())
    #         print(f"  Max SHAPE: {max_tensor.squeeze().cpu().numpy().shape}")
    #         print("Max min : ", max_tensor.min())
    #         print("Max max : ", max_tensor.max())

    layers_name = {}
    layers_name[intermediate_bounds_list[0]] = 0
    for k in range(1, K + 1):
        layers_name[intermediate_bounds_list[1 + (k - 1) * 3]] = k

    print("Intermediate bounds list final values : ", intermediate_bounds_list[-1])
    layers_name[intermediate_bounds_list[-1]] = K

    for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():

        if layer_name not in layers_name:
            # print(f"Layer {layer_name} not found in layers_name mapping.")
            # print(f"  Min: {min_tensor.squeeze().shape}")
            # print(f"  Max: {max_tensor.squeeze().shape} \n")
            continue
        # print(f"{layer_name}:")
        # print(f"  Min: {min_tensor.squeeze().shape}")
        # print(f"  Max: {max_tensor.squeeze().shape} \n")
        if layers_name[layer_name] == 0:
            # For the first layer, we set the lower bound to 0
            min_tensor = torch.clamp(min_tensor, min=0).view(-1)
            max_tensor = max_tensor.view(-1)

        L[layers_name[layer_name]] = (
            min_tensor.squeeze().detach().cpu().numpy().tolist()
        )
        U[layers_name[layer_name]] = (
            max_tensor.squeeze().detach().cpu().numpy().tolist()
        )

    L = round_list_depth_2(L)
    U = round_list_depth_2(U)
    
    return L, U


def compute_bounds(self, method: str = "IBP"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    start_compute_bd_time = time.time()
    L, U = compute_bounds_data(
        self.network, self.x, self.epsilon, self.n, self.K, method=method
    )
    end_compute_bd_time = time.time()
    self.compute_bounds_time = end_compute_bd_time - start_compute_bd_time
    self.L = L
    self.U = U


def check_stability_neurons(
    self, use_active_neurons: bool = False, use_inactive_neurons: bool = False
):
    """
    Check the stability of neurons in the network.
    """
    print("STUDY : Checking stability of neurons ...")
    self.stable_inactives_neurons = []
    self.stable_actives_neurons = []
    # Check if the neurons are stable
    for k in range(1, self.K):
        for j in range(self.n[k]):
            if self.L[k][j] <= 0 and self.U[k][j] <= 0 and not use_inactive_neurons:
                self.stable_inactives_neurons.append((k, j))
            elif self.L[k][j] >= 0 and self.U[k][j] > 0 and not use_active_neurons:
                self.stable_actives_neurons.append((k, j))
    self.stable_active_neurons = set(self.stable_actives_neurons)
    self.stable_inactive_neurons = set(self.stable_inactives_neurons)
    print(
        "STUDY : Stable neurons : ",
        len(self.stable_active_neurons) + len(self.stable_inactive_neurons),
    )


def prune_adversarial_targets(self):
    """
    Prune the adversarial targets based on the bounds : targets with upper bound lower than other target's lower bound is removed from the adversarial target.
    """
    for j in list(self.ytargets):

        if j == self.ytrue:
            continue
        if self.U[self.K][j] <= self.L[self.K][self.ytrue]:
            self.ytargets.remove(j)
        elif any(
            self.U[self.K][j] < self.L[self.K][j2] for j2 in self.ytargets if j2 != j
        ):
            self.ytargets.remove(j)
        else:
            # print("STUDY : Adversarial target selected : ", j)
            continue
