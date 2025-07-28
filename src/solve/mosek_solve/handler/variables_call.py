import numpy as np

from .indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from .indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)
from .variable_elements import (
    Equivalent_Neurons_Index,
    Equivalent_Betas_Index,
    _get_linear_indices_from_key,
    _get_key_from_layer_neuron_,
    _get_layer_neuron_from_key_,
    _get_key_linear_,
)
import logging
from typing import List
from collections import Counter
from numba import njit
import numba
from numba.typed import Dict

logger_mosek = logging.getLogger("Mosek_logger")

from tools import summing_values_two_dicts, change_to_zero_negative_values


def get_only_one_variable_kwargs(index: int = 1, **kwargs):
    assert index in [1, 2], "Index must be either 1 or 2."
    return {
        k.strip(f"{index}"): v for k, v in kwargs.items() if str(k).endswith(f"{index}")
    }


class LayersValues:
    """
    A class to handle the values of neurons (in particular stable active neurons) accros layers
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        W: list,
        b: list,
        stable_inactives_neurons: List[tuple] = [],
        stable_actives_neurons: List[tuple] = [],
        L: List[List[float]] = None,
        U: List[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize the LayersValues class.
        """
        self.n = n
        self.W = W
        self.b = b
        self.K = K
        self.stable_inactives_neurons = stable_inactives_neurons
        self.stable_actives_neurons = stable_actives_neurons
        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.keep_actives_penultimate = kwargs.get("keep_penultimate_actives", None)
        assert (
            self.keep_actives_penultimate is not None
        ), "keep_penultimate_actives must be specified."

        self.equivalent_values_layers = {
            (layer, neuron): {"neurons_weight": {}, "constant": 0}
            for layer in range(K + 1)
            for neuron in range(n[layer])
        }

        print(" K in LayersValues:", self.K)
        print("len equivalent_values_layers:", len(self.equivalent_values_layers))
        for k in range(K + 1):
            for j in range(n[k]):
                self.add_equivalent_values(k, j)

    def add_equivalent_values(self, layer: int, neuron: int):
        if (
            ((layer, neuron) in self.stable_actives_neurons)
            and not (self.keep_actives_penultimate and layer == self.K - 1)
        ) or (layer == self.K and not self.LAST_LAYER):
            for i in range(self.n[layer - 1]):
                # equivalent_neuron_i_weights = self.equivalent_values_layers[layer - 1][i]["neurons_weight"]
                # equivalent_neuron_i_constant = self.equivalent_values_layers[layer - 1][i]["constant"]

                self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = (
                    summing_values_two_dicts(
                        self.equivalent_values_layers[(layer, neuron)][
                            "neurons_weight"
                        ],
                        {
                            (layer2, neuron2): (value * self.W[layer - 1][neuron][i])
                            for (
                                layer2,
                                neuron2,
                            ), value in self.equivalent_values_layers[(layer - 1, i)][
                                "neurons_weight"
                            ].items()
                        },
                    )
                )
                self.equivalent_values_layers[(layer, neuron)]["constant"] += (
                    self.equivalent_values_layers[(layer - 1, i)]["constant"]
                    * self.W[layer - 1][neuron][i]
                )

            self.equivalent_values_layers[(layer, neuron)]["constant"] += self.b[
                layer - 1
            ][neuron]

            coordinates = [
                (layer, neuron)
                for (layer, neuron), value in self.equivalent_values_layers[
                    (layer, neuron)
                ]["neurons_weight"].items()
            ]
            counts = Counter(coordinates)

        elif (layer, neuron) in self.stable_inactives_neurons:
            pass
        elif layer == self.K and self.LAST_LAYER:
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = {
                (layer, neuron): 1
            }

        else:
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = {
                (layer, neuron): 1
            }

    def get_equivalent_values(self, layer: int, neuron: int):
        """
        Get the equivalent values for a given layer and neuron.
        """
        if layer < 0 or layer > self.K:
            raise ValueError(f"Layer {layer} is out of bounds (0 to {self.K}).")
        if neuron < 0 or neuron >= self.n[layer]:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is out of bounds (0 to {self.n[layer] - 1})."
            )
        return (
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"],
            self.equivalent_values_layers[(layer, neuron)]["constant"],
        )

    def is_unstable(self, layer: int, neuron: int) -> bool:
        """
        Check if the neuron is a stable active neuron.
        """
        return (layer, neuron) not in (
            self.stable_actives_neurons + self.stable_inactives_neurons
        ) and (layer is not None and neuron is not None)

    def is_stable_active(self, layer: int, neuron: int) -> bool:
        """
        Check if the neuron is a stable active neuron.
        """
        return (layer, neuron) in self.stable_actives_neurons and (
            layer is not None and neuron is not None
        )

    def computing_bounds_based_on_stable_neurons(
        self,
        L: List[List[float]] = None,
        U: List[List[float]] = None,
    ):
        """
        Compute the bounds based on stable neurons.
        """
        for k in range(self.K + 1):
            for j in range(self.n[k]):
                if (k, j) in self.stable_actives_neurons:
                    upper_bounds = (
                        sum(
                            value * U[k][j]
                            for (k, j), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value > 0
                        )
                        + sum(
                            value * L[k][j]
                            for (k, j), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value < 0
                        )
                        + self.equivalent_values_layers[k, j]["constant"]
                    )
                    if upper_bounds < U[k][j]:
                        print(
                            "The computed upper bound is BETTER on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial upper bound.",
                        )
                        U[k][j] = upper_bounds
                    else:
                        print(
                            "The computed upper bound is not better on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial upper bound.",
                        )
                    # print(f"Upper bound for layer {k}, neuron {j}: {self.upper_bounds[(k, j)]} and U = {U[k][j]}")
                    lower_bounds = (
                        sum(
                            value * L[k][j]
                            for (k, j), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value > 0
                        )
                        + sum(
                            value * U[k][j]
                            for (k, j), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value < 0
                        )
                        + self.equivalent_values_layers[k, j]["constant"]
                    )
                    if lower_bounds > L[k][j]:
                        print(
                            "The computed lower bound is BETTER on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial lower bound.",
                        )
                        L[k][j] = lower_bounds
                    else:
                        print(
                            "The computed lower bound is not better on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial lower bound.",
                        )
                    # print(f"Lower bound for layer {k}, neuron {j}: {self.lower_bounds[(k, j)]} and L = {L[k][j]}")
        return L, U


# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************


class VariablesCall:
    """
    A class to handle the variables call in the MOSEK solver.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        """
        Initialize the VariablesCall class.
        """
        self.indexes_matrices = indexes_matrices
        self.indexes_variables = indexes_variables

        self.stable_inactives_neurons = kwargs.pop("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.pop("stable_actives_neurons")
        self.ytargets = kwargs.get("ytargets")

        self.K = kwargs.pop("K")
        self.n = kwargs.pop("n")
        self.W = kwargs.pop("W")
        self.b = kwargs.pop("b")

        self.L = kwargs.get("L", None)
        self.U = kwargs.get("U", None)

        print(
            "maxabs(U) : ", max([max([abs(U_i_j) for U_i_j in U_i])] for U_i in self.U)
        )
        print(
            "maxabs(L) : ", max([max([abs(L_i_j) for L_i_j in L_i])] for L_i in self.L)
        )

        print("Starting Layers Values initialization with K:", self.K)
        self.layers_values = LayersValues(
            K=self.K,
            n=self.n,
            W=self.W,
            b=self.b,
            stable_actives_neurons=self.stable_actives_neurons,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )
        print("Layers Values initialized with K:", self.K)
        print("self.n in VariablesCall:", self.n)

        self.L, self.U = self.layers_values.computing_bounds_based_on_stable_neurons(
            L=self.L, U=self.U
        )
        self.U_above_zero = change_to_zero_negative_values(
            self.U, dim=2
        )  # ATTENTION U N'EST PAS PRECIS : POUR EVITER CAS DES NEURONES STABLES INACTIFS
        self.L_above_zero = change_to_zero_negative_values(
            self.L, dim=2
        )  # ATTENTION : CECI POSERA UN PROBLEME POUR LES CONTRAINTES TRIANGULAIRES

        self.LAST_LAYER = kwargs.get("LAST_LAYER", None)
        self.BETAS = kwargs.get("BETAS", None)
        assert self.LAST_LAYER is not None, "LAST_LAYER must be specified."
        assert self.BETAS is not None, "BETAS must be specified."

        self.equivalent_neurons = Equivalent_Neurons_Index()
        self.equivalent_indexes_betas = Equivalent_Betas_Index(ytargets=self.ytargets)
        self.create_equivalent_indexes_matrices()
        self._print_equivalent_indexes_()
        self.study_indexes_equivalent_neurons()

    def create_equivalent_indexes_matrices(self):

        for layer in range(self.K + 1):
            for neuron in range(self.n[layer]):

                equivalent_values_neurons, constant = (
                    self.layers_values.get_equivalent_values(layer, neuron)
                )
                self.equivalent_neurons.create_dict(layer=layer, neuron=neuron)
                for (k, j), val in equivalent_values_neurons.items():

                    if (k < self.K - 1 and not self.LAST_LAYER) or (
                        k < self.K and self.LAST_LAYER
                    ):
                        ind_i = self.indexes_variables._get_variable_index(
                            "z", layer=k, neuron=j, front_of_matrix=True
                        )
                        ind_num_matrix = self.indexes_matrices._get_matrix_index(
                            "z", layer=k, neuron=j, front_of_matrix=True
                        )
                        self.equivalent_neurons.add(
                            layer=layer,
                            neuron=neuron,
                            i=ind_i,
                            num_matrix=ind_num_matrix,
                            value=val,
                            front_of_matrix=True,
                        )
                    if k > 0:
                        ind_i_back = self.indexes_variables._get_variable_index(
                            "z", layer=k, neuron=j, front_of_matrix=False
                        )
                        ind_num_matrix_back = self.indexes_matrices._get_matrix_index(
                            "z", layer=k, neuron=j, front_of_matrix=False
                        )
                        self.equivalent_neurons.add(
                            layer=layer,
                            neuron=neuron,
                            i=ind_i_back,
                            num_matrix=ind_num_matrix_back,
                            value=val,
                            front_of_matrix=False,
                        )

                self.equivalent_neurons.add_constant(
                    layer=layer,
                    neuron=neuron,
                    value=constant,
                )

        if self.BETAS:
            print(
                "Adding equivalent indexes for betas" " for class labels:",
                self.ytargets,
            )

            for class_label in self.ytargets:

                i = self.indexes_variables._get_variable_index(
                    "beta", class_label=class_label
                )
                num_matrix = self.indexes_matrices._get_matrix_index(
                    "beta", class_label=class_label
                )
                self.equivalent_indexes_betas.add(
                    class_label=class_label,
                    i=i,
                    num_matrix=num_matrix,
                )
            print("Equivalent indexes for betas added.")
            print(self.equivalent_indexes_betas.equivalent_indexes_betas)

    def study_indexes_equivalent_neurons(self):
        for k in range(self.K + 1):
            for j in range(self.n[k]):
                key = self.equivalent_neurons.get_index(layer=k, neuron=j)
                layer, neuron = _get_layer_neuron_from_key_(key=key)
                assert (
                    layer == k and neuron == j
                ), f"Error in STUDY1: layer = {layer}, neuron = {neuron}, k = {k}, j = {j}"
        for num_matrix in range(self.indexes_matrices.nb_matrices):
            try:
                for i in range(self.indexes_variables.max_index):

                    key = _get_key_linear_(i, num_matrix)

                    i2, num_matrix2 = _get_linear_indices_from_key(key)

                    assert (
                        i == i2 and num_matrix == num_matrix2
                    ), f"Error in STUDY2: i = {i}, num_matrix = {num_matrix}, i2 = {i2}, num_matrix2 = {num_matrix2}"

            except ValueError as e:
                print("Error : ", e)
                pass

    def _print_equivalent_indexes_(self):
        line = ""

        for layer in range(self.K + 1):
            line += f"Layer {layer}:\n"
            for neuron in range(self.n[layer]):
                front = self.equivalent_neurons.get_equivalent(
                    layer=layer, neuron=neuron, front_of_matrix=True
                )
                back = self.equivalent_neurons.get_equivalent(
                    layer=layer, neuron=neuron, front_of_matrix=False
                )
                constant = self.equivalent_neurons.get_constant(
                    layer=layer, neuron=neuron
                )
                line += f"\n  Layer {layer} Neuron {neuron}: \n"
                line += f"    FRONT_OF_MATRIX \n"
                for key, value in front.items():
                    i, num_matrix = _get_linear_indices_from_key(key, 13)
                    line += f"  {(num_matrix,i)} : {value};   "
                line += f"\n    BACK_OF_MATRIX \n"
                for key, value in back.items():
                    i, num_matrix = _get_linear_indices_from_key(key, 13)
                    line += f"{(num_matrix,i)} : {value};   "
                line += f"    constant : {constant}\n"
        if self.BETAS:
            for class_label in self.ytargets:
                line += f"Class {class_label}:"
                dict_beta = self.equivalent_indexes_betas.get_equivalent(class_label)
                for key, value in dict_beta.items():
                    i, num_matrix = _get_linear_indices_from_key(key, 13)
                    line += f"  i : {i} ; "
                    line += f"  num_matrix : {num_matrix}\n"
        print(line)

        ""

    def add_constant(self, value: float):
        """
        Add a constant to the constraint.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def add_var(self, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_upper_bound_value(self, layer: int, neuron: int) -> float:
        """
        Get the upper bound value for a given layer and neuron.
        """
        if layer < 0 or layer > self.K:
            raise ValueError(f"Layer {layer} is out of bounds (0 to {self.K}).")
        if neuron < 0 or neuron >= self.n[layer]:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is out of bounds (0 to {self.n[layer] - 1})."
            )
        return self.layers_values.upper_bounds.get((layer, neuron))

    def get_lower_bound_value(self, layer: int, neuron: int) -> float:
        """
        Get the upper bound value for a given layer and neuron.
        """
        if layer < 0 or layer > self.K:
            raise ValueError(f"Layer {layer} is out of bounds (0 to {self.K}).")
        if neuron < 0 or neuron >= self.n[layer]:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is out of bounds (0 to {self.n[layer] - 1})."
            )
        return self.layers_values.lower_bounds.get((layer, neuron))

    def call_variable(self, var: str, **kwargs):
        pass

    def add_linear_variable(self, var: str, value: float, **kwargs):
        """
        Add a linear variables to the constraint.
        Checks if the variable present corresponds to stable active neurons and divides it in this case to the precedent layer's z variables.
        """
        if value == 0:
            return
        if var == "z":
            layer = kwargs.get("layer", None)
            neuron = kwargs.get("neuron", None)
            front_of_matrix = kwargs.get("front_of_matrix", True)
            assert layer is not None, "Layer must be specified for z variable."
            assert neuron is not None, "Neuron must be specified for z variable."
            assert (
                front_of_matrix is not None
            ), "Front of matrix must be specified for z variable."

            self.add_var(
                dict1=self.equivalent_neurons.get_equivalent(
                    layer=layer, neuron=neuron, front_of_matrix=front_of_matrix
                ),
                value=value,
            )

            constant = self.equivalent_neurons.get_constant(layer, neuron)
            self.add_constant(value * constant)
        else:
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."

            self.add_var(
                dict1=self.equivalent_indexes_betas.get_equivalent(
                    class_label=class_label
                ),
                value=value,
            )

    def add_quad_variable(self, var1: str, var2: str, value: float, **kwargs):
        """
        Add a product of two variables to the constraint.
        Checks if the variables present corresponds to stable active neurons and divides them in this case to the precedent layer's z variables.
        """
        if value == 0:
            return
        assert var1 in ["z", "beta"], "var1 must be either 'z' or 'beta'."
        assert var2 in ["z", "beta"], "var2 must be either 'z' or 'beta'."
        # Special case for product of two z unstable neurons variables
        # print(
        #     "Adding quad variable with var1:",
        #     var1,
        #     "var2:",
        #     var2,
        #     "value:",
        #     value,
        #     "kwargs : ",
        #     kwargs,
        # )
        if var1 == "z" and var2 == "beta":
            layer1 = kwargs.get("layer1", None)
            neuron1 = kwargs.get("neuron1", None)
            front_of_matrix1 = kwargs.get("front_of_matrix1", True)
            assert layer1 is not None, "Layer must be specified for z variable."
            assert neuron1 is not None, "Neuron must be specified for z variable."
            assert (
                front_of_matrix1 is not None
            ), "Front of matrix must be specified for z variable."
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."
            dict2 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label
            )
            dict1 = self.equivalent_neurons.get_equivalent(
                layer1, neuron1, front_of_matrix1
            )
            constant1 = self.equivalent_neurons.get_constant(layer1, neuron1)
            constant2 = 0

        elif var1 == "beta" and var2 == "z":
            layer2 = kwargs.get("layer2", None)
            neuron2 = kwargs.get("neuron2", None)
            front_of_matrix2 = kwargs.get("front_of_matrix2", True)
            assert layer2 is not None, "Layer must be specified for z variable."
            assert neuron2 is not None, "Neuron must be specified for z variable."
            assert (
                front_of_matrix2 is not None
            ), "Front of matrix must be specified for z variable."
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."
            dict1 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label
            )
            dict2 = self.equivalent_neurons.get_equivalent(
                layer2, neuron2, front_of_matrix2
            )

            constant1 = 0
            constant2 = self.equivalent_neurons.get_constant(layer2, neuron2)

        elif var1 == "beta" and var2 == "beta":
            class_label1 = kwargs.get("class_label1", None)
            class_label2 = kwargs.get("class_label2", None)
            assert (
                class_label1 is not None
            ), "Class label1 must be specified for beta variable."
            assert (
                class_label2 is not None
            ), "Class label2 must be specified for beta variable."
            dict1 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label1
            )
            dict2 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label2
            )
            constant1 = 0
            constant2 = 0

        elif var1 == "z" and var2 == "z":
            layer1 = kwargs.get("layer1", None)
            neuron1 = kwargs.get("neuron1", None)
            front_of_matrix1 = kwargs.get("front_of_matrix1", True)
            assert layer1 is not None, "Layer must be specified for z variable."
            assert neuron1 is not None, "Neuron must be specified for z variable."
            assert (
                front_of_matrix1 is not None
            ), "Front of matrix must be specified for z variable."
            layer2 = kwargs.get("layer2", None)
            neuron2 = kwargs.get("neuron2", None)
            front_of_matrix2 = kwargs.get("front_of_matrix2", True)
            assert layer2 is not None, "Layer must be specified for z variable."
            assert neuron2 is not None, "Neuron must be specified for z variable."
            assert (
                front_of_matrix2 is not None
            ), "Front of matrix must be specified for z variable."
            constant1 = self.equivalent_neurons.get_constant(layer1, neuron1)
            constant2 = self.equivalent_neurons.get_constant(layer2, neuron2)
            dict1 = self.equivalent_neurons.get_equivalent(
                layer=layer1, neuron=neuron1, front_of_matrix=front_of_matrix1
            )
            dict2 = self.equivalent_neurons.get_equivalent(
                layer=layer2, neuron=neuron2, front_of_matrix=front_of_matrix2
            )

        self.add_var(
            dict1=dict1,
            value=value,
            dict2=dict2,
        )
        if constant1 != 0 and constant2 != 0:
            self.add_constant(value * constant1 * constant2)
        if constant1 != 0:
            self.add_var(dict1=dict2, value=value * constant1)
        if constant2 != 0:
            self.add_var(dict1=dict1, value=value * constant2)

    # def add_linear_variable(self, var: str, value: float, **kwargs):
    #     """
    #     Add a linear variables to the constraint.
    #     Checks if the variable present corresponds to stable active neurons and divides it in this case to the precedent layer's z variables.
    #     """
    #     print(
    #         "\n \n \n           Adding linear variable:",
    #         var,
    #         "with value:",
    #         value,
    #         "and kwargs:",
    #         kwargs,
    #     )
    #     if value == 0:
    #         print("Value is 0, skipping addition of variable.")
    #         return
    #     i, num_matrix, val, constant = self.call_variable(var, **kwargs)
    #     j = np.zeros(i.shape, dtype=int)  # Assuming j is always 0 for linear variables
    #     self.add_var(
    #         i=i,
    #         j=j,
    #         num_matrix=num_matrix,
    #         value=value * val,
    #     )
    #     self.add_constant(value * constant)

    # def add_quad_variable(self, var1: str, var2: str, value: float, **kwargs):
    #     """
    #     Add a product of two variables to the constraint.
    #     Checks if the variables present corresponds to stable active neurons and divides them in this case to the precedent layer's z variables.
    #     """
    #     print(
    #         "\n \n          Adding quadratic variable:",
    #         var1,
    #         "and",
    #         var2,
    #         "with value:",
    #         value,
    #         "and kwargs:",
    #         kwargs,
    #     )
    #     if value == 0:
    #         print("Value is 0, skipping addition of variable.")
    #         return
    #     if var1 == "z":
    #         layer = kwargs.get("layer1", None)
    #         neuron = kwargs.get("neuron1", None)
    #         front_of_matrix = kwargs.get("front_of_matrix1", True)
    #         assert layer is not None, "Layer must be specified for z variable."
    #         assert neuron is not None, "Neuron must be specified for z variable."
    #         assert (
    #             front_of_matrix is not None
    #         ), "Front of matrix must be specified for z variable."
    #         i1, num_matrix1, val1, constant1 = self.call_variable(
    #             var1, layer=layer, neuron=neuron, front_of_matrix=front_of_matrix
    #         )

    #     else:
    #         class_label = kwargs.get("class_label", None)
    #         assert (
    #             class_label is not None
    #         ), "Class label must be specified for beta variable."
    #         i1, num_matrix1, val1, constant1 = self.call_variable(
    #             var1, class_label=class_label
    #         )

    #     if var2 == "z":
    #         layer = kwargs.get("layer2", None)
    #         neuron = kwargs.get("neuron2", None)
    #         front_of_matrix = kwargs.get("front_of_matrix2", True)
    #         assert layer is not None, "Layer must be specified for z variable."
    #         assert neuron is not None, "Neuron must be specified for z variable."
    #         assert (
    #             front_of_matrix is not None
    #         ), "Front of matrix must be specified for z variable."
    #         i2, num_matrix2, val2, constant2 = self.call_variable(
    #             var2, layer=layer, neuron=neuron, front_of_matrix=front_of_matrix
    #         )

    #     else:
    #         class_label = kwargs.get("class_label", None)
    #         assert (
    #             class_label is not None
    #         ), "Class label must be specified for beta variable."
    #         i2, num_matrix2, val2, constant2 = self.call_variable(
    #             var2, class_label=class_label
    #         )

    #     print("num_matrix1 :", num_matrix1, "num_matrix2 :", num_matrix2)
    #     assert (
    #         set(num_matrix1) == set(num_matrix2) and len(set(num_matrix1)) == 1
    #     ), "The matrices for the two variables must be the same and should have only one value."

    #     i1_broadcast = i1[:, np.newaxis]
    #     i2_broadcast = i2[np.newaxis, :]
    #     val1_broadcast = val1[:, np.newaxis]
    #     val2_broadcast = val2[np.newaxis, :]

    #     val_quad = np.multiply(val1_broadcast, val2_broadcast).flatten()
    #     print("val :", val_quad)
    #     i_j = np.array(np.meshgrid(i2_broadcast, i1_broadcast)).flatten()
    #     i_j = np.array(i_j).flatten()
    #     print("val .shape :     ", val_quad.shape)
    #     i_quad = i_j[: val_quad.shape[0]]
    #     j_quad = i_j[val_quad.shape[0] :]
    #     print("i :", i_quad)
    #     print("j : ", j_quad)
    #     num_matrix = np.broadcast_to(
    #         num_matrix1[:, np.newaxis], (len(num_matrix1), len(num_matrix2))
    #     ).flatten()
    #     print("num_matrix :", num_matrix)

    #     self.add_var(i=i_quad, j=j_quad, num_matrix=num_matrix, value=val_quad * value)

    #     if constant1 != 0 or constant2 != 0:
    #         self.add_constant(value * (constant1 + constant2))
    #     if constant1 != 0:
    #         self.add_var(
    #             i=i2,
    #             j=np.zeros(i2.shape, dtype=int),
    #             num_matrix=num_matrix2,
    #             value=val2 * value * constant1,
    #         )
    #     if constant2 != 0:
    #         self.add_var(
    #             i=i1,
    #             j=np.zeros(i1.shape, dtype=int),
    #             num_matrix=num_matrix1,
    #             value=val1 * value * constant2,
    #         )
