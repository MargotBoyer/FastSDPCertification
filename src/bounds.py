import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time

from networks import network
from tools import round_list_depth_2, change_to_zero_negative_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_bounds_data(network, x, epsilon, n, K, method: str = "IBP", norm : str = "Linf"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    print(f"STUDY : Computing bounds with method: {method} ...")
    print("epsilon : ", epsilon)
    L = [[-np.inf] * n[k] for k in range(K + 1)]
    U = [[np.inf] * n[k] for k in range(K + 1)]

    if method == "GREAT_BOUNDS":
        L[0] = [max(L[0][j], 0) for j in range(len(L[0]))]
        return

    if not torch.is_tensor(x):
        x = torch.Tensor(x)

    x = x.type(torch.float).view(-1).unsqueeze(0).to(device)
    print("x device : ", x.device)
    print("x shape : ", x.shape)

    network = network.to(device)
    print("network device : ", next(network.parameters()).device)
    network.eval()
    print("network is none : ", network is None)

    zeros = torch.zeros_like(x).to(device)
    print("zeros device : ", zeros.device)

    print("STUDY : creating BoundedModule ...")
    try:

        # # Vérif optional : assure que tout est bien sur cuda
        # for param in network.parameters():
        #     print("param :  ", param)
        #     print("device :  ", device)
        #     print("param device :  ", param.device)
        #     assert param.device == device
        # print('parameters are on the right device')
        # for buf in network.buffers():
        #     assert buf.device == device
        # print('buffers are on the right device')
        print("About to create BoundedModule on device:", device)
        print("network device before BoundedModule:", next(network.parameters()).device)
        print("zeros device before BoundedModule:", zeros.device)
        bounded_model = BoundedModule(
            network,
            zeros,
            bound_opts={"conv_mode": "patches"},
        )
        print("created BoundedModule")
    except Exception as e:
        print("Error creating BoundedModule:", e)
        return

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    # # S'assurer que le modèle est bien déplacé
    # network.to(device)

    # # Vérifier que tous les paramètres ET buffers sont bien sur le même device
    # for name, param in network.named_parameters():
    #     print(f"STUDY : Parameter {name} is on device: {param.device}")

    # for name, layer in network.layers.items():
    #     print("STUDY : Layer : ", name)
    # # Créer l'entrée zéro sur le bon device
    # zeros = torch.zeros_like(x, device=device)

    # # Et maintenant seulement :
    # bounded_model = BoundedModule(
    #     network,
    #     zeros,
    #     bound_opts={"conv_mode": "patches"},
    # )

    bounded_model.eval()
    print("STUDY : bounded_model device : ", next(bounded_model.parameters()).device)

    if norm == "Linf":
        print("STUDY : Using Linf norm for perturbation.")
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    elif norm == "L2":
        print("STUDY : pertubation L2 used")
        ptb = PerturbationLpNorm(norm=2, eps=epsilon)
        #ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon**2)  # comparer les deux versions
    else:
        raise NotImplementedError(f"Norm {norm} not implemented.")
    bounded_image = BoundedTensor(x, ptb)
    if method == "alpha-CROWN":
        lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    else:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method=method)
    intermediate_bounds = bounded_model.save_intermediate()

    intermediate_bounds_list = list(intermediate_bounds.keys())
    print("STUDY : Intermediate bounds list : ", intermediate_bounds_list)

    for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():
        print(f"STUDY : {layer_name}")
        print(f"STUDY : shape = {max_tensor.squeeze().detach().cpu().numpy().shape}")
        
        # if self.data_modele == "blob":
        #     print(f"  Min: {min_tensor.squeeze().cpu().numpy()}")
        #     print(f"  Max: {max_tensor.squeeze().cpu().numpy()}")
        # else:
        # print(f"  Min SHAPE: {min_tensor.squeeze().cpu().numpy().shape}")
        print("STUDY : Min min : ", min_tensor.min())
        # print("Min max : ", min_tensor.max())
        # print(f"  Max SHAPE: {max_tensor.squeeze().cpu().numpy().shape}")
        # print("Max min : ", max_tensor.min())
        print("STUDY : Max max : ", max_tensor.max())

    layers_name = {}
    layers_name[intermediate_bounds_list[0]] = 0

    print("STUDY : Preparing to create bounds...")
    print('Intermediate_bounds_list : ', intermediate_bounds_list )
    for k in range(1, K + 1):
        print(f"Adding layer for k = {k}, num_layer = {1 + (k - 1) * 2}")
        layers_name[intermediate_bounds_list[1 + (k - 1) * 2]] = k    ### !!!!  Before *3 because of the dropout layer  !!!!
    print("STUDY : Layers name mapping : ", layers_name)

    print("STUDY : Intermediate bounds list final values : ", intermediate_bounds_list[-1])
    layers_name[intermediate_bounds_list[-1]] = K

    for layer_name, (min_tensor, max_tensor) in intermediate_bounds.items():

        if layer_name not in layers_name:
            print(f"Layer {layer_name} not found in layers_name mapping.")
            print(f"  Min: {min_tensor.squeeze().shape}")
            print(f"  Max: {max_tensor.squeeze().shape} \n")
            continue
        print(f"{layer_name}:")
        print(f"  Min: {min_tensor.squeeze().shape}")
        print(f"  Max: {max_tensor.squeeze().shape} \n")
        if layers_name[layer_name] == 0:
            # For the first layer, we set the lower bound to 0
            min_tensor = torch.clamp(min_tensor, min=0).view(-1)
            max_tensor = max_tensor.view(-1)

        print(
            f"STUDY : Adding layer : {layer_name} : {layers_name[layer_name]}, min = {min_tensor.min().item()}, max = {max_tensor.max().item()}"
        )

        L[layers_name[layer_name]] = (
            min_tensor.squeeze().detach().cpu().numpy().tolist()
        )
        U[layers_name[layer_name]] = (
            max_tensor.squeeze().detach().cpu().numpy().tolist()
        )

    L = round_list_depth_2(L)
    U = round_list_depth_2(U)

    # for k in range(len(L)):
    #     min_layer_diff = 1e10
    #     max_layer_diff = -1e10
    #     min_layer_diff_ecart_relatif = 1e10
    #     min_layer = min(L[k])
    #     max_layer = max(U[k])
    #     for j in range(len(L[k])):
    #         if L[k][j] > U[k][j]:
    #             print(
    #                 f"STUDY : Warning: Inconsistent bounds at layer {k}, neuron {j}: L={L[k][j]} > U={U[k][j]}. Adjusting L to U."
    #             )
    #         else :
    #             if U[k][j] - L[k][j] < min_layer_diff:
    #                 min_layer_diff = U[k][j] - L[k][j]
    #             if U[k][j] - L[k][j] > max_layer_diff:
    #                 max_layer_diff = U[k][j] - L[k][j]
    #             if 2*(U[k][j] - L[k][j]) / (abs(U[k][j]) + abs(L[k][j])) < min_layer_diff_ecart_relatif:
    #                 min_layer_diff_ecart_relatif = 2*(U[k][j] - L[k][j]) / (abs(U[k][j]) + abs(L[k][j]))

    #     print("STUDY : Bounds differences at layer ", k, " : min=", min_layer, ";  max=", max_layer, " : min_diff=", min_layer_diff, ";  max_diff=", max_layer_diff, ";  rel_min=", min_layer_diff_ecart_relatif)

    return L, U


def compute_bounds(self, method: str = "IBP"):
    """
    Compute the  L and U

    Args:
        method (str): The method to compute the bounds (CROWN, IBP, Linear, etc.).
    """
    print("STUDY : Computing bounds with norm: ", self.norm, " ...")
    start_compute_bd_time = time.time()
    L, U = compute_bounds_data(
        self.network, self.x, self.epsilon, self.n, self.K, method=method, norm=self.norm
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
            # print(
            #     "STUDY : Layer ",
            #     k,
            #     " Neuron ",
            #     j,
            #     " L=",
            #     self.L[k][j],
            #     " U=",
            #     self.U[k][j],
            # )
            if self.L[k][j] <= 0 and self.U[k][j] <= 0 and not use_inactive_neurons:
                self.stable_inactives_neurons.append((k, j))
            elif self.L[k][j] >= 0 and self.U[k][j] > 0 and not use_active_neurons:
                self.stable_actives_neurons.append((k, j))
    self.stable_active_neurons = set(self.stable_actives_neurons)
    self.stable_inactive_neurons = set(self.stable_inactives_neurons)
    print(
        "STUDY : Nb Stable neurons : ",
        len(self.stable_active_neurons) + len(self.stable_inactive_neurons),
    )
    print("STUDY : stable active neurons : ", self.stable_active_neurons)
    print("STUDY : stable inactive neurons : ", self.stable_inactive_neurons)


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
