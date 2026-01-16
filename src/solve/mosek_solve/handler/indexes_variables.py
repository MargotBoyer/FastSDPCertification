from typing import List


class Indexes_Variables_for_Mosek_Solver:
    """
    Class to handle the indexes of the variables and constraints in the MOSEK solver.
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        ytrue: int,
        MATRIX_BY_LAYERS: bool = False,
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        BETAS_Z: bool = False,
        ZBAR: bool = False,
        **kwargs,
    ):
        """
        Initialize the Indexes_Mosek_Solver class.

        Parameters
        ----------
        n: List[int]
            List of the number of neurons in each layer.
        K: int
            Number of layers.
        matrix_by_layers: bool
            Whether to use matrix by layers or not.
        last_layer: bool
            Whether the last layer is included in the matrix of the z variables or not.
        betas: bool
            Whether to include the beta variables or not.
        betas_z: bool
            Whether to include the beta variables in the matrixes for z variables.
        zbar: bool
            Whether to include the zbar variables or not.
        """
        self.n = n
        self.K = K
        self.MATRIX_BY_LAYERS = MATRIX_BY_LAYERS
        self.keep_penultimate_actives = kwargs.get("keep_penultimate_actives", False)
        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR
        self.ytrue = ytrue

        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.get("stable_actives_neurons")
        self.ytargets = kwargs.get("ytargets")
        self.count_max_indexes()

    def get_number_pruned_neurons_on_layer(self, layer: int, neuron: int = None) -> int:
        """
        Get the number of inactive neurons on a given layer and before a given neuron.

        Parameters
        ----------
        layer: int
            The layer number.
        neuron: int
            The neuron number.

        Returns
        -------
        int
            The number of inactive neurons.
        """
        if (
            self.stable_inactives_neurons is None
            and self.stable_actives_neurons is None
        ):
            return 0

        if neuron is None:
            neuron = self.n[layer]

        if (layer == self.K - 1) and self.keep_penultimate_actives:
            return len(
                [
                    (k, j)
                    for k, j in (self.stable_inactives_neurons)
                    if (k == layer and j < neuron)
                ]
            )
        else:
            return len(
                [
                    (k, j)
                    for k, j in (
                        self.stable_inactives_neurons + self.stable_actives_neurons
                    )
                    if (k == layer and j < neuron)
                ]
            )

    def count_max_indexes(self) -> int:
        """
        Count the maximum index based on the configuration.
        """
        max_index = 0
        if self.MATRIX_BY_LAYERS:
            for k in range(self.K if self.LAST_LAYER else self.K - 1):
                max_index_k = 1 + self.n[k] + self.n[k + 1]
                if k == self.K - 1 and not self.LAST_LAYER and self.BETAS_Z:
                    max_index_k += len(self.ytargets)
                elif k == self.K and self.LAST_LAYER:
                    max_index_k += len(self.ytargets) + 1
                if max_index_k > max_index:
                    max_index = max_index_k

        else:
            if self.LAST_LAYER:
                max_index = sum(self.n) + 1
            else:
                max_index = sum(self.n[: self.K]) + 1
            if self.BETAS_Z:
                max_index += len(self.ytargets)
        if not self.BETAS_Z and self.BETAS:
            if len(self.ytargets) > max_index:
                max_index = len(self.ytargets) + 1

        self.max_index = max_index + 1000

    def get_number_pruned_neurons_before_layer(
        self, layer: int, neuron: int = None
    ) -> int:
        """
        Get the number of inactive neurons before a given layer and neuron.

        Parameters
        ----------
        layer: int
            The layer number.
        neuron: int
            The neuron number.

        Returns
        -------
        int
            The number of inactive neurons.
        """
        if self.stable_inactives_neurons is None:
            return 0

        if neuron is None:
            neuron = self.n[layer]
        return sum(
            self.get_number_pruned_neurons_on_layer(k) for k in range(layer)
        ) + self.get_number_pruned_neurons_on_layer(layer, neuron)

    def index_variable_z(self, layer: int, neuron: int, front_of_matrix: bool) -> int:
        """
        Get the index of the z variable for a given layer and neuron.

        Parameters
        ----------
        layer: int
            The layer number.
        neuron: int
            The neuron number.

        Returns
        -------
        int
            The index of the z variable.
        """
        if (layer == self.K and not self.LAST_LAYER) or layer < 0 or layer > self.K:
            raise ValueError(f"Layer index {layer} out of range.")
        if (layer, neuron) in self.stable_inactives_neurons:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is inactive and has no z variable."
            )
        if self.MATRIX_BY_LAYERS:
            if front_of_matrix is True:
                if (layer == self.K - 1 and not self.LAST_LAYER) or (
                    layer == self.K and self.LAST_LAYER
                ):
                    raise ValueError(
                        f"Layer {layer} can not be at the front of the matrix."
                    )
                else:
                    return (
                        1
                        + neuron
                        - self.get_number_pruned_neurons_on_layer(layer, neuron)
                    )
            else:
                if layer == 0:
                    raise ValueError("Layer 0 can not be at the back of the matrix.")
                elif layer == self.K:
                    assert self.LAST_LAYER
                    return (
                        1
                        + self.n[layer - 1]
                        + neuron
                        - self.get_number_pruned_adversarial_targets_before_target(
                            neuron
                        )
                        - self.get_number_pruned_neurons_on_layer(layer - 1)
                        - self.get_number_pruned_neurons_on_layer(layer, neuron)
                    )
                else:
                    return (
                        1
                        + self.n[layer - 1]
                        + neuron
                        - self.get_number_pruned_neurons_on_layer(layer - 1)
                        - self.get_number_pruned_neurons_on_layer(layer, neuron)
                    )
        else:
            if self.LAST_LAYER:
                return (
                    1
                    + sum(self.n[:layer])
                    + neuron
                    - self.get_number_pruned_adversarial_targets_before_target(neuron)
                    - self.get_number_pruned_neurons_before_layer(layer, neuron)
                )

            else:
                return (
                    1
                    + sum(self.n[:layer])
                    + neuron
                    - self.get_number_pruned_neurons_before_layer(layer, neuron)
                )

    def get_number_pruned_adversarial_targets_before_target(self, ytarget) -> int:
        """
        Get the number of pruned adversarial targets before the true target.

        Returns
        -------
        int
            The number of pruned adversarial targets.
        """
        return len(
            [
                class_label
                for class_label in range(self.n[self.K])
                if (
                    class_label != self.ytrue
                    and class_label not in self.ytargets
                    and class_label < ytarget
                )
            ]
        )

    def ind_label_beta(self, class_label: int) -> int:
        """
        Get the index of the beta variable for a given class label.
        Function to be called in the index_variable_beta function.

        Parameters
        ----------
        class_label: int
            The class label.

        Returns
        -------
        int
            The index of the beta variable in the list of target classes.
        """
        if self.BETAS:
            if class_label < self.ytrue:
                return (
                    class_label
                    - self.get_number_pruned_adversarial_targets_before_target(
                        class_label
                    )
                )
            elif class_label == self.ytrue:
                raise ValueError("The true class label has no beta variable.")
            else:
                return (
                    class_label
                    - 1
                    - self.get_number_pruned_adversarial_targets_before_target(
                        class_label
                    )
                )

        else:
            raise ValueError("Beta variables are not enabled.")

    def index_variable_beta(self, class_label: int) -> int:
        """
        Get the index of the beta variable for a given class label.

        Parameters
        ----------
        class_label: int
            The class label.

        Returns
        -------
        int
            The index of the beta variable.
        """
        assert self.BETAS

        if self.BETAS_Z:
            if self.MATRIX_BY_LAYERS:
                if self.LAST_LAYER:
                    if self.ZBAR:
                        return (
                            1
                            + self.n[self.K - 1]
                            + len(self.ytargets)
                            + 1
                            + 1
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_on_layer(self.K - 1)
                            - self.get_number_pruned_neurons_on_layer(self.K)
                        )
                    else:
                        return (
                            1
                            + self.n[self.K - 1]
                            + len(self.ytargets)
                            + 1
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_on_layer(self.K - 1)
                            - self.get_number_pruned_neurons_on_layer(self.K)
                        )
                else:
                    if self.ZBAR:

                        return (
                            1
                            + self.n[self.K - 2]
                            + self.n[self.K - 1]
                            + 1
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_on_layer(self.K - 2)
                            - self.get_number_pruned_neurons_on_layer(self.K - 1)
                        )
                    else:
                        return (
                            1
                            + self.n[self.K - 2]
                            + self.n[self.K - 1]
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_on_layer(self.K - 2)
                            - self.get_number_pruned_neurons_on_layer(self.K - 1)
                        )
            else:
                if self.LAST_LAYER:
                    if self.ZBAR:
                        return (
                            1
                            + sum(self.n)
                            + 1
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_before_layer(self.K + 1)
                        )
                    else:
                        return (
                            1
                            + sum(self.n)
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_before_layer(self.K + 1)
                        )
                else:
                    if self.ZBAR:
                        return (
                            1
                            + sum(self.n[: self.K])
                            + 1
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_before_layer(self.K)
                        )
                    else:
                        return (
                            1
                            + sum(self.n[: self.K])
                            + self.ind_label_beta(class_label)
                            - self.get_number_pruned_neurons_before_layer(self.K)
                        )
        else:
            return 1 + self.ind_label_beta(class_label)

    def index_variable_zbar(self) -> int:
        """
        Get the index of the zbar variable.

        Returns
        -------
        int
            The index of the zbar variable.
        """
        assert self.ZBAR
        assert self.BETAS_Z

        if self.MATRIX_BY_LAYERS:
            if self.LAST_LAYER:
                return (
                    1
                    + self.n[self.K - 1]
                    + len(self.ytargets)
                    + 1
                    - self.get_number_pruned_neurons_on_layer(self.K - 1)
                    - self.get_number_pruned_neurons_on_layer(self.K)
                )
            else:
                return (
                    1
                    + self.n[self.K - 2]
                    + self.n[self.K - 1]
                    - self.get_number_pruned_neurons_on_layer(self.K - 2)
                    - self.get_number_pruned_neurons_on_layer(self.K - 1)
                )
        else:
            if self.LAST_LAYER:
                return (
                    1
                    + sum(self.n)
                    - self.get_number_pruned_neurons_before_layer(self.K + 1)
                )
            else:
                return (
                    1
                    + sum(self.n[: self.K])
                    - self.get_number_pruned_neurons_before_layer(self.K)
                )

    def _get_variable_index(self, var_type: str, is_first: bool = None, **kwargs):
        """
        Helper method to get variable index based on type and parameters for z, beta, and zbar variables.
        Parameters :
        var_type : str
            Type of the variable (z, beta, zbar)
        is_first : bool
            Whether the variable is the first or second in the pair (True for linear variables)
        """
        if is_first is None:
            suffix = ""
        else:
            suffix = "1" if is_first else "2"
        front_of_matrix = kwargs.get(f"front_of_matrix{suffix}", None)



        if var_type == "z":
            # For z variables
            layer = kwargs.get(
                f"layer{suffix}" if f"layer{suffix}" in kwargs else "layer"
            )
            neuron = kwargs.get(
                f"neuron{suffix}" if f"neuron{suffix}" in kwargs else "neuron"
            )

            if front_of_matrix is None and (layer == self.K and self.LAST_LAYER):
                front_of_matrix = False
            elif front_of_matrix is None and (
                layer == self.K - 1 and not self.LAST_LAYER
            ):
                front_of_matrix = False
            elif front_of_matrix is None:
                front_of_matrix = True
            if layer is None or neuron is None:
                raise ValueError(
                    f"Layer and neuron required for z variable ({'first' if is_first else 'second'})"
                )

            return self.index_variable_z(layer, neuron, front_of_matrix)

        elif var_type == "beta":
            # For beta variables
            class_label = kwargs.get(
                f"class_label{suffix}"
                if f"class_label{suffix}" in kwargs
                else "class_label"
            )

            if class_label is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )
            return self.index_variable_beta(class_label)

        elif var_type == "zbar":
            # For zbar variables
            return self.index_variable_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")
