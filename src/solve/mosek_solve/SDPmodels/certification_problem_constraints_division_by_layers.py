import mosek


# *********************************************RELU ***************************************************************
def matrix_by_layers_rec(self, only_linear_constraints: bool = False):
    print("Adding rec matrices constraint")
    sum_cstr = 0
    for k in range(1, self.K if self.LAST_LAYER else self.K - 1):
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                continue
            elif (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                continue
            # P_{k-1}[z_{k,j}] == P_{k}[z_{k,j}]
            if self.handler.Constraints.new_constraint(
                f"Rec: P_{k-1}[z_{k,j}] == P_{k}[z_{k,j}]"
            ):
                return
            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
                front_of_matrix=True,
            )
            self.handler.Constraints.add_linear_variable(
                "z",
                value=-1,
                layer=k,
                neuron=j,
                front_of_matrix=False,
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.fx,
                bound=0,
            )
            sum_cstr += 1

    if not only_linear_constraints:
        for k in range(1, self.K if self.LAST_LAYER else self.K - 1):
            for j in range(self.n[k]):
                if (k, j) in self.stable_inactives_neurons:
                    continue
                elif (k, j) in self.stable_actives_neurons and (
                    not self.keep_penultimate_actives or k != self.K - 1
                ):
                    continue
                for j2 in range(j + 1):
                    if (k, j2) in self.stable_inactives_neurons :
                        continue
                    elif (k, j2) in self.stable_actives_neurons and (
                        not self.keep_penultimate_actives or k != self.K - 1
                    ):
                        continue
                    # P_{k-1}[z_{k,j} * z_{k,j2}] == P_{k}[z_{k,j} * z_{k,j2}]
                    if self.handler.Constraints.new_constraint(
                        f"Rec: P_{k-1}[z_{k,j}  * z_{k,j2}] == P_{k}[z_{k,j} *  * z_{k,j2}]"
                    ):
                        return
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=k,
                        neuron1=j,
                        var2="z",
                        layer2=k,
                        neuron2=j2,
                        value=1,
                        front_of_matrix1=True,
                        front_of_matrix2=True,
                    )
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=k,
                        neuron1=j,
                        var2="z",
                        layer2=k,
                        neuron2=j2,
                        value=-1,
                        front_of_matrix1=False,
                        front_of_matrix2=False,
                    )

                    self.handler.Constraints.add_bound(
                        bound_type=mosek.boundkey.fx,
                        bound=0,
                    )

                    sum_cstr += 1
    print(
        f"Number of constraints for the matrix by layers: {sum_cstr} "
        f"for {self.K} layers and {self.n} neurons"
    )
