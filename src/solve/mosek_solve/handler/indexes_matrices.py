from typing import List


class Indexes_Matrixes_for_Mosek_Solver:
    """
    Class to handle the indexes of the variables and constraints in the MOSEK solver.
    """

    def __init__(
        self,
        K: int,
        n: List[int],
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
        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR

        self.ytargets = kwargs.get("ytargets")

        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.get("stable_actives_neurons")
        self.check_conformity()

        self.current_matrices_variables = []
        self.count_nb_matrices()

    def count_nb_matrices(self):
        """
        Count the number of matrices based on the configuration.
        """
        self.nb_matrices = 0
        if self.MATRIX_BY_LAYERS:
            self.nb_matrices += self.K - 1
            if self.LAST_LAYER:
                self.nb_matrices += 1
            if not self.BETAS_Z and self.BETAS:
                self.nb_matrices += 1
        else:
            self.nb_matrices += 1
            if not self.BETAS_Z and self.BETAS:
                self.nb_matrices += 1

    def check_conformity(self):
        """
        Check the presence of unstable neurons in each layer. (TO MAKE : extension for layers with no unstable neurons)
        Check the number of layers and the number of neurons in each layer.
        """

        assert self.K == len(self.n) - 1
        # Each layer has unstable neurons
        if self.MATRIX_BY_LAYERS:

            for layer in range(self.K, 2):
                taille = len(
                    [
                        (layer, j)
                        for j in range(self.n[layer])
                        if (layer, j) not in self.stable_inactives_neurons
                        and (layer, j) not in self.stable_actives_neurons
                    ]
                    + [
                        (layer, j)
                        for j in range(self.n[layer + 1])
                        if (layer, j) not in self.stable_inactives_neurons
                        and (layer, j) not in self.stable_actives_neurons
                    ]
                )
                print("Taille de la matrice pour la couche ", layer, " : ", taille)
                assert (
                    len(
                        [
                            (layer, j)
                            for j in range(self.n[layer])
                            if (layer, j) not in self.stable_inactives_neurons
                            and (layer, j) not in self.stable_actives_neurons
                        ]
                        + [
                            (layer, j)
                            for j in range(self.n[layer + 1])
                            if (layer, j) not in self.stable_inactives_neurons
                            and (layer, j) not in self.stable_actives_neurons
                        ]
                    )
                    > 0
                )
        if any(
            all(
                (layer, j) in self.stable_inactives_neurons
                for j in range(self.n[layer])
            )
            for layer in range(1, self.K)
        ):
            raise ValueError(
                "There are layers with only inactive neurons : a special treatment is needed : the output is fixed."
            )

    def is_in_matrix_with_betas(self, layer: int) -> bool:
        """
        Check if the matrix for beta variables is used in the given layer.

        Parameters
        ----------
        layer: int
            The layer number to check.

        Returns
        -------
        bool
            True if the matrix for beta variables is used in the given layer, False otherwise.
        """
        if self.BETAS_Z:
            if self.MATRIX_BY_LAYERS:
                if self.LAST_LAYER:
                    return layer == self.K - 1 or layer == self.K
                else:
                    return layer == self.K - 2 or layer == self.K - 1
            else:
                return True
        else:
            return False

    def index_matrix_z(self, layer: int, front_of_matrix: bool) -> int:
        """
        Get the index of the matrix variable for the variable z for a given layer and neuron.

        Parameters
        ----------
        layer: int
            The layer number.
        front_of_matrix: bool
            Whether the variable is at the front of the matrix or not.
        Returns
        -------
        int
            The index of the matrix for the z variable.
        """

        if (layer == self.K and not self.LAST_LAYER) or layer < 0 or layer > self.K:
            raise ValueError(f"Layer index {layer} out of range.")
        if self.MATRIX_BY_LAYERS:
            if front_of_matrix:
                if (layer == self.K - 1 and not self.LAST_LAYER) or (
                    layer == self.K and self.LAST_LAYER
                ):
                    raise ValueError(
                        f"Layer {layer} can not be at the front of the matrix."
                    )
                else:
                    return layer
            else:
                if layer == 0:
                    raise ValueError("Layer 0 can not be at the back of the matrix.")
                else:
                    return layer - 1
        else:
            return 0

    def index_matrix_beta(self) -> int:
        """
        Get the index of the matrix with the beta variables.
        """
        assert self.BETAS
        if self.BETAS_Z:
            if self.MATRIX_BY_LAYERS:
                if self.LAST_LAYER:
                    return self.K - 1
                else:
                    return self.K - 2
            else:
                return 0
        else:
            if self.MATRIX_BY_LAYERS:
                if self.LAST_LAYER:
                    return self.K
                else:
                    return self.K - 1
            else:
                return 1

    def index_matrix_zbar(self) -> int:
        """
        Get the index of the matrix of the zbar variable.

        Returns
        -------
        int
            The index of the matrix with the zbar variable.
        """
        assert self.ZBAR
        assert self.BETAS_Z

        if self.MATRIX_BY_LAYERS:
            if self.LAST_LAYER:
                return self.K - 1
            else:
                return self.K - 2
        else:
            return 0

    def _get_matrix_index(self, var_type: str, is_first: bool = None, **kwargs):
        """
        Helper method to get matrix index based on type and parameters for z, beta, and zbar variables.
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
            if front_of_matrix is None and (layer == self.K and self.LAST_LAYER):
                front_of_matrix = False
            elif front_of_matrix is None and (
                layer == self.K - 1 and not self.LAST_LAYER
            ):
                front_of_matrix = False
            elif front_of_matrix is None:
                front_of_matrix = True

            if layer is None:
                raise ValueError(
                    f"Layer required for z variable ({'first' if is_first else 'second'})"
                )

            return self.index_matrix_z(layer, front_of_matrix)

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

            return self.index_matrix_beta()

        elif var_type == "zbar":
            # For zbar variables
            return self.index_matrix_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def get_shape_matrix(self, num_matrix: int):
        """
        Get the shape of the matrix based on the number of neurons in each layer.
        """
        return self.current_matrices_variables[num_matrix]["dim"]

    def get_name_matrix(self, num_matrix: int):
        """
        Get the name of the matrix based on the number of neurons in each layer.
        """
        return self.current_matrices_variables[num_matrix]["name"]
