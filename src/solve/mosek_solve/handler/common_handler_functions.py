def print_index_variables_matrices(self):
    """
    String representation of the class.
    """
    line = ""

    for layer in range(self.K + 1 if self.LAST_LAYER else self.K):
        line += f"\n Layer {layer} : \n"

        for j in range(self.n[layer]):
            line += f"      Neuron {j} : \n"

            if (layer, j) in self.stable_inactives_neurons:
                line += "           is inactive \n"
                continue
            if (layer < self.K - 1 and not self.LAST_LAYER) or (
                layer < self.K and self.LAST_LAYER
            ):

                ind_matrix_front = self.indexes_matrices._get_matrix_index(
                    "z", layer=layer, neuron=j, front_of_matrix=True
                )

                ind_col_front = self.indexes_variables._get_variable_index(
                    "z", layer=layer, neuron=j, front_of_matrix=True
                )
                line += f"          front : index = {ind_matrix_front}, i = {ind_col_front} \n"
            if layer > 0:
                ind_col_back = self.indexes_variables._get_variable_index(
                    "z", layer=layer, neuron=j, front_of_matrix=False
                )
                ind_matrix_back = self.indexes_matrices._get_matrix_index(
                    "z", layer=layer, neuron=j, front_of_matrix=False
                )
                line += f"          back  : index = {ind_matrix_back}, i = {ind_col_back} \n"

    if self.ZBAR:
        line += "\n  Zbar : \n"
        ind_matrix = self.indexes_matrices._get_matrix_index("zbar")
        ind_col = self.indexes_variables._get_variable_index("zbar")
        line += f"          index = {ind_matrix}, i = {ind_col} \n"

    if self.BETAS:
        line += "\n  Betas : \n"
        for class_label in self.ytargets:
            if class_label == self.ytrue:
                continue
            line += f"      Class {class_label} : \n"
            ind_matrix = self.indexes_matrices._get_matrix_index(
                "beta", class_label=class_label
            )
            ind_col = self.indexes_variables._get_variable_index(
                "beta", class_label=class_label
            )
            line += f"          index = {ind_matrix}, i = {ind_col} \n"

    print(line)
    return line


def num_matrices_variables(self):
    """
    Return the number of matrices.
    """
    return len(self.indexes_matrices.current_matrices_variables)

def print_num_variables(self):
    num_variables = 0
    for i in range(self.num_matrices_variables()):
        dim = self.indexes_matrices.current_matrices_variables[i]["dim"]
        num_variables += (dim+1) * (dim+1)
    print(f"CALLBACK num variables : {num_variables}")
