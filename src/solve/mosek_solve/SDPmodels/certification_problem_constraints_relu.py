import mosek
from tools import infinity
import logging

logger_mosek = logging.getLogger("Mosek_logger")


def ReLU_constraint_stable_active_relaxation(
    self,
    k,
    j,
    upper_bound_neuron: bool = True,
    lower_bound_neuron: bool = True,
    upper_bound_decomposed: bool = True,
    lower_bound_decomposed: bool = True,
):

    assert any(
        (k - 1, i) in self.stable_actives_neurons for i in range(self.n[k - 1])
    ), f"Neuron ({k}, {j}) has no previous stable active neuron."

    if upper_bound_decomposed:
        if self.handler.Constraints.new_constraint(
            f"ReLU - z_{k,j} * (z{k,j} - W_{k,j}' z_{k-1}' - b_{k,j}) - U_or_L{k,j} * W_{k,j}'' * z_{k-1}'' <= 0"
        ):
            return

        self.handler.Constraints.add_quad_variable(
            var1="z",
            layer1=k,
            neuron1=j,
            var2="z",
            layer2=k,
            neuron2=j,
            value=1,
            front_of_matrix1=False,
            front_of_matrix2=False,
        )

        self.handler.Constraints.add_linear_variable(
            "z",
            value=-self.network.b[k - 1][j],
            layer=k,
            neuron=j,
            front_of_matrix=False,
        )

        for i in range(self.n[k - 1]):
            if (k - 1, i) in self.stable_inactives_neurons:
                continue
            elif (k - 1, i) in self.stable_actives_neurons:
                weighted_neurons, constant = (
                    self.handler.Constraints.layers_values.get_equivalent_values(
                        k - 1, i
                    )
                )
                for (layer, neuron), val in weighted_neurons.items():
                    if val * self.network.W[k - 1][j][i] > 0:
                        self.handler.Constraints.add_linear_variable(
                            "z",
                            value=-self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.U[k][j],
                            layer=layer,
                            neuron=neuron,
                        )
                    else:
                        self.handler.Constraints.add_linear_variable(
                            "z",
                            value=-self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.L[k][j],
                            layer=layer,
                            neuron=neuron,
                        )
                if self.network.W[k - 1][j][i] * constant > 0:
                    self.handler.Constraints.add_constant(
                        value=-self.network.W[k - 1][j][i]
                        * constant
                        * self.handler.Constraints.U[k][j]
                    )
                else:
                    self.handler.Constraints.add_constant(
                        value=-self.network.W[k - 1][j][i]
                        * constant
                        * self.handler.Constraints.L[k][j]
                    )

            else:
                # print(f"Adding non-stable neuron ({k-1}, {i}) with weight {self.network.W[k - 1][j][i]}")
                self.handler.Constraints.add_quad_variable(
                    var1="z",
                    layer1=k,
                    neuron1=j,
                    var2="z",
                    layer2=k - 1,
                    neuron2=i,
                    value=-self.network.W[k - 1][j][i],
                    front_of_matrix1=False,
                    front_of_matrix2=True,
                )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)

    if lower_bound_decomposed:
        if self.handler.Constraints.new_constraint(
            f"ReLU - z_{k,j} * (z{k,j} - W_{k,j}' z_{k-1}' - b_{k,j}) - U_or_L{k,j} * W_{k,j}'' * z_{k-1}'' >= 0"
        ):
            return

        self.handler.Constraints.add_quad_variable(
            var1="z",
            layer1=k,
            neuron1=j,
            var2="z",
            layer2=k,
            neuron2=j,
            value=1,
            front_of_matrix1=False,
            front_of_matrix2=False,
        )

        self.handler.Constraints.add_linear_variable(
            "z",
            value=-self.network.b[k - 1][j],
            layer=k,
            neuron=j,
            front_of_matrix=False,
        )

        for i in range(self.n[k - 1]):
            if (k - 1, i) in self.stable_inactives_neurons:
                continue
            elif (k - 1, i) in self.stable_actives_neurons:
                weighted_neurons, constant = (
                    self.handler.Constraints.layers_values.get_equivalent_values(
                        k - 1, i
                    )
                )
                for (layer, neuron), val in weighted_neurons.items():
                    if val * self.network.W[k - 1][j][i] > 0:
                        self.handler.Constraints.add_linear_variable(
                            "z",
                            value=-self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.L[k][j],
                            layer=layer,
                            neuron=neuron,
                        )
                    else:
                        self.handler.Constraints.add_linear_variable(
                            "z",
                            value=-self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.U[k][j],
                            layer=layer,
                            neuron=neuron,
                        )
                if self.network.W[k - 1][j][i] * constant > 0:
                    self.handler.Constraints.add_constant(
                        value=-self.network.W[k - 1][j][i]
                        * constant
                        * self.handler.Constraints.L[k][j]
                    )
                else:
                    self.handler.Constraints.add_constant(
                        value=-self.network.W[k - 1][j][i]
                        * constant
                        * self.handler.Constraints.U[k][j]
                    )

            else:
                # print(f"Adding non-stable neuron ({k-1}, {i}) with weight {self.network.W[k - 1][j][i]}")
                self.handler.Constraints.add_quad_variable(
                    var1="z",
                    layer1=k,
                    neuron1=j,
                    var2="z",
                    layer2=k - 1,
                    neuron2=i,
                    value=-self.network.W[k - 1][j][i],
                    front_of_matrix1=False,
                    front_of_matrix2=True,
                )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.lo, bound=0)

    if upper_bound_neuron:
        if self.handler.Constraints.new_constraint(
            f"ReLU - z_{k,j} * (z{k,j} - W_{k,j}' z_{k-1}' - b_{k,j}) - M_{k,j} * z_{k,j}'' <= 0"
        ):
            return

        self.handler.Constraints.add_quad_variable(
            var1="z",
            layer1=k,
            neuron1=j,
            var2="z",
            layer2=k,
            neuron2=j,
            value=1,
            front_of_matrix1=False,
            front_of_matrix2=False,
        )

        self.handler.Constraints.add_linear_variable(
            "z",
            value=-self.network.b[k - 1][j],
            layer=k,
            neuron=j,
            front_of_matrix=False,
        )

        M_up = 0
        for i in range(self.n[k - 1]):
            if (k - 1, i) in self.stable_inactives_neurons:
                continue
            elif (k - 1, i) in self.stable_actives_neurons:
                weighted_neurons, constant = (
                    self.handler.Constraints.layers_values.get_equivalent_values(
                        k - 1, i
                    )
                )
                for (layer, neuron), val in weighted_neurons.items():
                    if val * self.network.W[k - 1][j][i] > 0:
                        M_up += (
                            self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.U[layer][neuron]
                        )
                    else:
                        M_up += (
                            self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.L[layer][neuron]
                        )
                M_up += self.network.W[k - 1][j][i] * constant
            else:
                # print(f"Adding non-stable neuron ({k-1}, {i}) with weight {self.network.W[k - 1][j][i]}")
                self.handler.Constraints.add_quad_variable(
                    var1="z",
                    layer1=k,
                    neuron1=j,
                    var2="z",
                    layer2=k - 1,
                    neuron2=i,
                    value=-self.network.W[k - 1][j][i],
                    front_of_matrix1=False,
                    front_of_matrix2=True,
                )

        self.handler.Constraints.add_linear_variable(
            "z",
            value=-M_up,
            layer=k,
            neuron=j,
            front_of_matrix=False,
        )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)

    if lower_bound_neuron:
        if self.handler.Constraints.new_constraint(
            f"ReLU - z_{k,j} * (z{k,j} - W_{k,j}' z_{k-1}' - b_{k,j}) - M_{k,j} * z_{k,j}'' <= 0"
        ):
            return

        self.handler.Constraints.add_quad_variable(
            var1="z",
            layer1=k,
            neuron1=j,
            var2="z",
            layer2=k,
            neuron2=j,
            value=1,
            front_of_matrix1=False,
            front_of_matrix2=False,
        )

        self.handler.Constraints.add_linear_variable(
            "z",
            value=-self.network.b[k - 1][j],
            layer=k,
            neuron=j,
            front_of_matrix=False,
        )

        M_lo = 0
        for i in range(self.n[k - 1]):
            if (k - 1, i) in self.stable_inactives_neurons:
                continue
            elif (k - 1, i) in self.stable_actives_neurons:
                weighted_neurons, constant = (
                    self.handler.Constraints.layers_values.get_equivalent_values(
                        k - 1, i
                    )
                )
                # print(f"Adding positive stable active neuron ({k-1}, {i}) with weight {self.network.W[k - 1][j][i]} and U = {self.handler.Constraints.U[k][j]}")
                for (layer, neuron), val in weighted_neurons.items():
                    if val * self.network.W[k - 1][j][i] > 0:
                        M_lo += (
                            self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.L[layer][neuron]
                        )
                    else:
                        M_lo += (
                            self.network.W[k - 1][j][i]
                            * val
                            * self.handler.Constraints.U[layer][neuron]
                        )
                M_lo += self.network.W[k - 1][j][i] * constant

            else:
                # print(f"Adding non-stable neuron ({k-1}, {i}) with weight {self.network.W[k - 1][j][i]}")
                self.handler.Constraints.add_quad_variable(
                    var1="z",
                    layer1=k,
                    neuron1=j,
                    var2="z",
                    layer2=k - 1,
                    neuron2=i,
                    value=-self.network.W[k - 1][j][i],
                    front_of_matrix1=False,
                    front_of_matrix2=True,
                )

        self.handler.Constraints.add_linear_variable(
            "z",
            value=-M_lo,
            layer=k,
            neuron=j,
            front_of_matrix=False,
        )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.lo, bound=0)


def ReLU_constraint_Lan(
    self,
    upper_bound_neuron: bool = True,
    lower_bound_neuron: bool = True,
    upper_bound_decomposed: bool = True,
    lower_bound_decomposed: bool = True,
):
    print("Adding quadratic RELU constraint")
    for k in range(1, self.K):
        print(f"Adding ReLU constraints for layer {k}")
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                # print(f"Skipping stable inactive neuron ({k}, {j})")
                continue
            if (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                # print(f"Skipping stable active neuron ({k}, {j})")
                continue
            # print(f"Adding ReLU constraint for layer {k}, neuron {j}")
            # zk >= 0
            if self.handler.Constraints.new_constraint(f"ReLU - z_{k,j}>=0"):
                continue
            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
                front_of_matrix=False,
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=0,
            )

            # zk >= Wk zk-1 + bk
            if self.handler.Constraints.new_constraint(
                f"ReLU - z_{k,j} >= W_{k,j} z_{k-1} + b{k,j}"
            ):
                continue

            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
                front_of_matrix=False,
            )

            for i in range(self.n[k - 1]):
                if (k - 1, i) in self.stable_inactives_neurons:
                    continue
                self.handler.Constraints.add_linear_variable(
                    "z",
                    value=-self.network.W[k - 1][j][i],
                    layer=k - 1,
                    neuron=i,
                )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=self.network.b[k - 1][j],
            )

            # zk * (zk - Wk zk-1 - bk) = 0
            if self.MATRIX_BY_LAYERS and any(
                (k - 1, i) in self.stable_actives_neurons for i in range(self.n[k - 1])
            ):
                # The constraint cannot be added as it links products of variables from different matrices : a relaxation is needed
                # print("Relaxation of ReLU constraint for layer", k, "neuron", j)
                self.ReLU_constraint_stable_active_relaxation(
                    k,
                    j,
                    upper_bound_neuron=upper_bound_neuron,
                    lower_bound_neuron=lower_bound_neuron,
                    upper_bound_decomposed=upper_bound_decomposed,
                    lower_bound_decomposed=lower_bound_decomposed,
                )

            else:
                # print("Adding normal ReLU constraint for layer", k, "neuron", j)
                if self.handler.Constraints.new_constraint(
                    f"ReLU - z_{k,j} * (z{k,j} - W_{k,j} z_{k-1} - b_{k,j}) = 0"
                ):
                    continue

                self.handler.Constraints.add_quad_variable(
                    var1="z",
                    layer1=k,
                    neuron1=j,
                    var2="z",
                    layer2=k,
                    neuron2=j,
                    value=1,
                    front_of_matrix1=False,
                    front_of_matrix2=False,
                )

                self.handler.Constraints.add_linear_variable(
                    "z",
                    value=-self.network.b[k - 1][j],
                    layer=k,
                    neuron=j,
                    front_of_matrix=False,
                )
                # Constraint can be added as it links products of variables from the same matrix
                for i in range(self.n[k - 1]):
                    if (k - 1, i) in self.stable_inactives_neurons:
                        continue
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=k,
                        neuron1=j,
                        var2="z",
                        layer2=k - 1,
                        neuron2=i,
                        value=-self.network.W[k - 1][j][i],
                        front_of_matrix1=False,
                        front_of_matrix2=True,
                    )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.fx, bound=0
                )


def ReLU_triangularization(self):
    for k in range(1, self.K + 1 if self.LAST_LAYER else self.K):
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                continue
            if (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                continue
            if (
                abs(self.handler.Constraints.U[k][j] - self.handler.Constraints.L[k][j])
                <= 1e-6
            ):
                logger_mosek.warning(
                    f"Layer {k}, Neuron {j} : L={self.handler.Constraints.L[k][j]} and U={self.handler.Constraints.U[k][j]} are equal, triangular ReLU constraint is not added."
                )
                continue

            rel_u = max(self.handler.Constraints.U[k][j], 0)
            rel_l = max(self.handler.Constraints.L[k][j], 0)
            k_cst = (rel_u - rel_l) / (
                self.handler.Constraints.U[k][j] - self.handler.Constraints.L[k][j]
            )
            # print(
            #     "k_cst:",
            #     k_cst,
            #     "rel_u:",
            #     rel_u,
            #     "rel_l:",
            #     rel_l,
            #     "L : ",
            #     self.handler.Constraints.L[k][j],
            #     "U :",
            #     self.handler.Constraints.U[k][j],
            # )

            # zk <= k * (Wk zk-1 + bk - Lk) + ReLU(Lk)
            if self.handler.Constraints.new_constraint(
                f"ReLU - z_{k,j} <= kcst * (W{k,j} z_{k-1} + b_{k,j} - L{k,j}) + ReLU(L_{k,j})"
            ):
                continue
            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
                front_of_matrix=False,
            )
            for i in range(self.n[k - 1]):
                if (k - 1, i) in self.stable_inactives_neurons:
                    continue
                self.handler.Constraints.add_linear_variable(
                    "z",
                    layer=k - 1,
                    neuron=i,
                    value=-k_cst * self.network.W[k - 1][j][i],
                )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.up,
                bound=rel_l
                + k_cst * (self.network.b[k - 1][j] - self.handler.Constraints.L[k][j]),
            )

            # self.handler.Constraints.print_current_constraint()
