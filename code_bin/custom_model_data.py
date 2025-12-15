#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
This file shows how to use customized models and customized dataloaders.

Use the example configuration:
python abcrown.py --config exp_configs/tutorial_examples/custom_model_data_example.yaml
"""

import os
import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
import arguments


def try_model_fsb(in_dim=784, out_dim=10):
    model = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def simple_conv_model(in_channel, out_dim):
    """Simple Convolutional model."""
    model = nn.Sequential(
        nn.Conv2d(in_channel, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, out_dim)
    )
    return model

def simple_fc_model_margot(K : int, n, in_dim=784, out_dim=10):
    """Simple Fully Connected model."""
    @torch.jit.ignore(drop=True)
    class ReLUNN(nn.Module):
        def __init__(self, K, n, W=None, b=None, dropout_prob: float = 0, name="ReLUNN"):

            super(ReLUNN, self).__init__()
            print("dropout prob debut : ", dropout_prob)
            self.name = name
            self.K = K
            self.W = W
            self.b = b
            self.n = n

            self.layers = nn.ModuleDict()
            self.penultimate_layer = "Layer_" + str(K - 1) + "_Linear"
            for k in range(1, K + 1):
                layer_lin = nn.Linear(in_features=self.n[k - 1], out_features=self.n[k])
                layer_lin_name = f"Layer_{k}_Linear"
                if W is not None:
                    layer_lin.weight = nn.Parameter(
                        torch.tensor(W[k - 1], dtype=torch.float32).clone().detach(),
                        requires_grad=False,
                    )
                    layer_lin.bias = nn.Parameter(
                        torch.tensor(b[k - 1], dtype=torch.float32).clone().detach(),
                        requires_grad=False,
                    )
                self.layers[layer_lin_name] = layer_lin

                if k < K:
                    layer_relu = nn.ReLU(inplace=True)
                    layer_relu_name = f"Layer_{k}_ReLU"
                    self.layers[layer_relu_name] = layer_relu
                    print("Dropout prob : ", dropout_prob)
                    self.layers["Layer_" + str(k) + "_Dropout"] = nn.Dropout(p=dropout_prob)
            if W is None:
                self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                # He initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def extract_weights(self):
            """
            Extrait les poids et biais des couches linéaires.
            Met à jour self.W et self.b avec les poids actuels.
            """
            W = []
            b = []

            for k in range(1, self.K + 1):
                layer_name = f"Layer_{k}_Linear"
                layer = self.layers[layer_name]
                W.append(layer.weight.data.tolist())
                b.append(layer.bias.data.tolist())

            self.W = W
            self.b = b
            return W, b

        def forward(self, x, verbose=False, return_last_hidden=False):
            """
            Forward pass through the network.
            Args:
                x (torch.Tensor): Input tensor.
                verbose (bool): If True, print intermediate outputs.
                return_last_hidden (bool): If True, return the output of the last hidden layer (penultimate layer).
                                        If False, return the final output.
            """
            # x = x.clone().detach().requires_grad_(True)

            if (
                x.dim() >= 4
            ):  # Si les données sont des images (batch_size, channels, height, width)
                x = x.view(x.size(0), -1)  # Aplatir les images (batch_size, 784)

            n_couche = 1
            for layer_name, layer in self.layers.items():

                x = layer(x)
                if (
                    layer_name == self.penultimate_layer
                ) and return_last_hidden:  # Avant-dernière couche (couche sans ReLU)
                    print("Derniere couche retournée")
                    return x

                if verbose:
                    if n_couche % 2 == 1:
                        print(f"Couche n° {n_couche//2} : ", x)
                    elif n_couche % 2 == 0:
                        print(f"Couche n° {n_couche//2} ReLU: ", x, " \n ")
                n_couche += 1

            return x

        def label(self, x):
            res = self.forward(x)
            label = torch.argmax(res)
            return label.item()

        def __str__(self, *args, **kwds):
            description = ""
            for layer_name, layer in self.layers.items():
                if isinstance(layer, nn.ReLU):
                    description += f"    {layer_name} ReLU : {layer}\n"
                elif isinstance(layer, nn.Dropout):
                    description += f"    {layer_name} Dropout : {layer}\n"
                else:
                    description += f"    {layer_name} Linear : {layer}\n"
                    description += f"    Weights : {layer.weight}\n"
                    description += f"    Bias : {layer.bias}\n"
                description += "\n"
            return description
    
    return ReLUNN(K, n)


def two_relu_toy_model(in_dim=2, out_dim=2):
    """A very simple model, 2 inputs, 2 ReLUs, 2 outputs"""
    model = nn.Sequential(
        nn.Linear(in_dim, 2),
        nn.ReLU(),
        nn.Linear(2, out_dim)
    )
    # [relu(x+2y)-relu(2x+y)+2, 0*relu(2x-y)+0*relu(-x+y)]
    model[0].weight.data = torch.tensor([[1., 2.], [2., 1.]])
    model[0].bias.data = torch.tensor([0., 0.])
    model[2].weight.data = torch.tensor([[1., -1.], [0., 0.]])
    model[2].bias.data = torch.tensor([2., 0.])
    return model


def simple_box_data(spec):
    """a customized box data: x=[-1, 1], y=[-1, 1]"""
    eps = spec["epsilon"]
    if eps is None:
        eps = 2.
    X = torch.tensor([[0., 0.]]).float()
    labels = torch.tensor([0]).long()
    eps_temp = torch.tensor(eps).reshape(1, -1)
    data_max = torch.tensor(10.).reshape(1, -1)
    data_min = torch.tensor(-10.).reshape(1, -1)
    return X, labels, data_max, data_min, eps_temp

def all_node_split_test_model(in_dim=2, out_dim=2):
    """A very simple model, 2 inputs, 20 ReLUs, 2 outputs"""
    model = nn.Sequential(
        nn.Linear(in_dim, 20),
        nn.ReLU(),
        nn.Linear(20, out_dim)
    )
    model[0].weight.data = torch.tensor([[-1.1258, -1.1524],
                                        [ 0.2506, -0.4339],
                                        [ 0.8487,  0.6920],
                                        [-0.3160, -2.1152],
                                        [ 0.4681, -0.1577],
                                        [ 1.4437,  0.2660],
                                        [ 0.1665,  0.8744],
                                        [-0.1435, -0.1116],
                                        [ 0.4736, -0.0729],
                                        [-0.8460,  0.1241],
                                        [ 0.2664,  0.4124],
                                        [-1.1480, -0.9625],
                                        [ 0.2343,  0.1264],
                                        [ 0.6591, -1.6591],
                                        [-1.0093, -1.4070],
                                        [ 0.2204, -0.1970],
                                        [-1.0683, -0.0390],
                                        [ 0.6933, -0.0684],
                                        [-0.5896,  0.7262],
                                        [ 0.8356, -0.1248]])
    model[0].bias.data = torch.tensor([-0.0043,  0.0017,  0.0020, -0.0005, -0.0030,  0.0011, -0.0029, -0.0023,  0.0037,  0.0023, -0.0025,  0.0041, -0.0082, -0.0077,  0.0006, -0.0022, -0.0045,  0.0003, -0.0033,  0.0020])
    model[2].weight.data = torch.tensor([[ 1.2026, -1.0299, -0.0809,  0.4990, -0.6472, -0.2247,  0.0726, -0.2912, -0.5695,  0.8674,
                                        -0.6774,  0.2767,  0.1709, -0.2701, -0.5633,  0.2803, -1.0325, -0.6330,  0.3569, -0.0638],
                                        [ 0.0129,  0.2553, -0.2982, -0.1459, -0.1255,  0.1057, -0.9055,  0.4570,  0.4074,  0.3204,
                                        -0.0127,  0.7773, -0.0831,  0.3661, -0.6250, -0.7922, -0.1339,  0.2914,  0.2083, -0.4933]])
    model[2].bias.data = torch.tensor([ 0.2484,  0.4397])
    return model


def box_data(dim, low=0., high=1., segments=10, num_classes=10, eps=None):
    """Generate fake datapoints."""
    step = (high - low) / segments
    data_min = torch.linspace(low, high - step, segments).unsqueeze(1).expand(segments, dim)  # Per element lower bounds.
    data_max = torch.linspace(low + step, high, segments).unsqueeze(1).expand(segments, dim)  # Per element upper bounds.
    X = (data_min + data_max) / 2.  # Fake data.
    labels = torch.remainder(torch.arange(0, segments, dtype=torch.int64), num_classes)  # Fake label.
    eps = None  # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    return X, labels, data_max, data_min, eps


def cifar10(spec, use_bounds=False):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    if use_bounds:
        # Option 1: for each example, we return its element-wise lower and upper bounds.
        # If you use this option, set --spec_type ("specifications"->"type" in config) to 'bound'.
        absolute_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        absolute_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        # Be careful with normalization.
        new_eps = torch.reshape(eps / std, (1, -1, 1, 1))
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        # In this case, the epsilon does not matter here.
        ret_eps = None
    else:
        # Option 2: return a single epsilon for all data examples, as well as clipping lower and upper bounds.
        # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        if eps is None:
            raise ValueError('You must specify an epsilon')
        # Rescale epsilon.
        ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps


def cifar10_filtered(spec):
    # Your existing cifar10 loader code...
    print("filter TEST TEST")
    X, labels, data_max, data_min, ret_eps = cifar10(spec)
    
    # Filter to only specific indices
    desired_indices = [10, 25, 30]
    X = X[desired_indices]
    labels = labels[desired_indices]
    
    return X, labels, data_max, data_min, ret_eps


def mnist_filtered(spec, use_bounds=False):
    eps = spec["epsilon"]
    print("epsilon in mnist_filtered:", eps)
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    train_data = datasets.MNIST(database_path, train=True, download=True, transform=transforms.ToTensor())
    # Load entire dataset.
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=10000, shuffle=False, num_workers=4)

    desired_indices = [10, 25, 30]

    X, labels = next(iter(dataloader))
    if use_bounds:
        # Option 1: for each example, we return its element-wise lower and upper bounds.
        # If you use this option, set --spec_type ("specifications"->"type" in config) to 'bound'.
        absolute_max = torch.tensor(1)
        absolute_min = torch.tensor(0)

        absolute_max = torch.reshape(absolute_max, (1, -1, 1, 1))
        absolute_min = torch.reshape(absolute_min, (1, -1, 1, 1))
        # Be careful with normalization.
        new_eps = eps
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        # In this case, the epsilon does not matter here.
        ret_eps = torch.reshape(eps, (1, -1, 1, 1))
    else :
        # Option 2: return a single epsilon for all data examples, as well as clipping lower and upper bounds.
        # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
        data_max = torch.tensor(1)
        data_min = torch.tensor(0)

        data_max = torch.reshape(data_max, (1, -1, 1, 1))
        data_min = torch.reshape(data_min, (1, -1, 1, 1))
        if eps is None:
            raise ValueError('You must specify an epsilon')
        # Rescale epsilon.
        eps = torch.tensor(eps)
        ret_eps = torch.reshape(eps, (1, -1, 1, 1))

    # X = X[desired_indices]
    # labels = labels[desired_indices]
    print("STUDY : X shape after filtering:", X.shape)
    print(f"STUDY : X shape after filtering: {X.shape}")
    print(f"STUDY : labels shape: {labels.shape}")
    print(f"STUDY : data_max shape: {data_max.shape if data_max is not None else None}")
    print(f"STUDY : data_min shape: {data_min.shape if data_min is not None else None}")
    print(f"STUDY : eps_temp shape : {ret_eps.shape}")
    return X, labels, data_max, data_min, ret_eps



def simple_cifar10(spec):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True,\
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data,\
            batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    if eps is None:
        raise ValueError('You must specify an epsilon')
    # Rescale epsilon.
    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps


def simple_cifar100(spec):
    """(Fait par Margot) Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    print("STUDY : simplecifar100 before")
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR100(database_path, train=True, download=True,\
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data,\
            batch_size=10000, shuffle=False, num_workers=4)
    print("STUDY : simplecifar100 after")
    X, labels = next(iter(testloader))
    # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    if eps is None:
        raise ValueError('You must specify an epsilon')
    # Rescale epsilon.
    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    print("STUDY : simplecifar100 before return")
    return X, labels, data_max, data_min, ret_eps

