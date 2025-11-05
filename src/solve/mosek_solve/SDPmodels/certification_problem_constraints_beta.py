import mosek
from tools import infinity


# *********************************************BETAS ***************************************************************
def discrete_betas(self):
    """
    Add the constraint betaj = betaj² for the beta variable (ensuring beta is in [0,1])
    """
    assert self.BETAS
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        # beta_j = beta_j²
        if self.handler.Constraints.new_constraint(
            f"beta_{j,j} = beta_{j}", label="same_for_data"
        ):
            continue
        self.handler.Constraints.add_quad_variable(
            var1="beta",
            class_label1=j,
            var2="beta",
            class_label2=j,
            value=1,
        )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=-1,
        )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)


def sum_betas_equals_1(self):
    """
    Add the constraint sum(betaj)=1 for the beta variable (ensuring beta is in [0,1])
    """
    assert self.BETAS

    if self.handler.Constraints.new_constraint("sum(beta)=1", label="same_for_data"):
        return
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=1,
        )
    self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=1)


def McCormick_beta_z(self, layer: int):
    if layer == self.K:
        assert self.LAST_LAYER
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        for i in range(self.n[layer]):
            if (layer, i) in self.stable_inactives_neurons:
                continue
            elif (layer, i) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or layer != self.K - 1
            ):
                continue

            front_of_matrix = (
                False
                if (
                    layer == self.K
                    and self.LAST_LAYER
                    or layer == self.K - 1
                    and not self.LAST_LAYER
                )
                else True
            )
            # *************************************************
            if self.handler.Constraints.new_constraint(
                f"T_{(layer, i),j}  <= U_{layer, i} beta_{j}", label="same_for_data"
            ):
                continue

            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=1,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j,
                value=-self.handler.Constraints.U_above_zero[layer][i],
            )
            self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)

            # ****************************************************
            if self.handler.Constraints.new_constraint(
                f"T_{(layer, i),j} <= z__{layer, i}"
            ):
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=1,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=layer,
                neuron=i,
                value=-1,
                front_of_matrix=front_of_matrix,
            )
            self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)

            # ****************************************************
            if self.handler.Constraints.new_constraint(
                f"T_{(layer, i),j}  >= U_{layer, i} beta_{j} + z_{layer, i} - U_{layer, i}",
                label="same_for_data",
            ):
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=1,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=layer,
                neuron=i,
                value=-1,
                front_of_matrix=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j,
                value=-self.handler.Constraints.U_above_zero[layer][i],
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=-self.handler.Constraints.U_above_zero[layer][i],
            )

            # ****************************************************
            if self.handler.Constraints.new_constraint(
                f"T_{(layer, i),j}  >= 0", label="same_for_data"
            ):
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=1,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_bound(bound_type=mosek.boundkey.lo, bound=0)


def McCormick_beta_z_with_penultimate_layer(self):
    """
    Add the McCormick constraints on zKj = WKj T_{(K-1, i)} + beta_j bKj
    """
    assert not self.LAST_LAYER
    assert self.BETAS and self.BETAS_Z
    for j in self.ytargets:
        if j == self.ytrue:
            continue

        if self.handler.Constraints.new_constraint(
            f"z_{self.K,j} beta_{j} >= L_{self.K,j} beta_{j}", label="same_for_data"
        ):
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=self.K - 1,
                neuron2=i,
                value=self.W[self.K - 1][j][i],
                front_of_matrix2=False,  # Always False for the last layer
            )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=self.b[self.K - 1][j],
        )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=-self.handler.Constraints.L[self.K][j],
        )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.lo, bound=0)

        # *************************************************
        if self.handler.Constraints.new_constraint(
            f"z_{self.K,j} beta_{j} <= U_{self.K,j} beta_{j}", label="same_for_data"
        ):
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=self.K - 1,
                neuron2=i,
                value=self.W[self.K - 1][j][i],
                front_of_matrix2=False,  # Always False for the last layer
            )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=self.b[self.K - 1][j],
        )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=-self.handler.Constraints.U_above_zero[self.K][j],
        )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)

        # *************************************************
        if self.handler.Constraints.new_constraint(
            f"z_{self.K,j} beta_{j} <= z_{self.K,j} + L_{self.K,j} beta_{j} - L_{self.K,j}",
            label="same_for_data",
        ):
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=self.K - 1,
                neuron2=i,
                value=self.W[self.K - 1][j][i],
                front_of_matrix2=False,  # Always False for the last layer
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K - 1,
                neuron=i,
                value=-self.W[self.K - 1][j][i],
                front_of_matrix=False,
            )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=self.b[self.K - 1][j],
        )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=-self.handler.Constraints.U_above_zero[self.K][j],
        )
        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.up,
            bound=self.b[self.K - 1][j] - self.handler.Constraints.L[self.K][j],
        )

        # *************************************************
        if self.handler.Constraints.new_constraint(
            f"z_{self.K,j} beta_{j} <= z_{self.K,j} + U_{self.K,j} beta_{j} - U_{self.K,j}",
            label="same_for_data",
        ):
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=self.K - 1,
                neuron2=i,
                value=self.W[self.K - 1][j][i],
                front_of_matrix2=False,  # Always False for the last layer
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K - 1,
                neuron=i,
                value=-self.W[self.K - 1][j][i],
                front_of_matrix=False,
            )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=self.b[self.K - 1][j],
        )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=self.handler.Constraints.U_above_zero[self.K][j],
        )
        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.lo,
            bound=self.b[self.K - 1][j]
            - self.handler.Constraints.U_above_zero[self.K][j],
        )


def betai_betaj(self):
    """
    Add the constraint betai * betaj
    """
    assert self.BETAS
    for j1 in self.ytargets:
        if j1 == self.ytrue:
            continue
        for j2 in self.ytargets:
            if j2 == self.ytrue or j2 >= j1:
                continue

            if self.BETAS_Z:
                # beta_j1 * beta_j2 = 0
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} * beta_{j2} = 0", label="same_for_data"
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.fx, bound=0
                )
            else:
                # beta_j1 * beta_j2 >= 0
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} * beta_{j2} >= 0", label="same_for_data"
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )

                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.lo, bound=0
                )

                # beta_j1 * beta_j2 >= betaj1 + betaj2 - 1
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} beta_{j2} >= beta_{j1} + beta_{j2} - 1",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j1,
                    value=-1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j2,
                    value=-1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.lo, bound=-1
                )

                # beta_j1 * beta_j2 <= min(betaj1, betaj2)
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} beta_{j2} <= min(beta_{j1}, beta_{j2})",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j1,
                    value=-1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.up, bound=0
                )

                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} beta_{j2} <= min(beta_{j1}, beta_{j2})",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j2,
                    value=-1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.up,
                    bound=0,
                )


def z_j2_beta_j2_greater_than_zj(self):
    """
    Add the constraint z_2 beta_2 >= z_1 - (1 - beta_2) U_1    (11)
    """
    assert self.BETAS
    assert not self.LAST_LAYER

    for j2 in self.ytargets:
        if j2 == self.ytrue:
            continue
        for j1 in self.ytargets:
            if j1 == self.ytrue or j1 == j2:
                continue
            if self.handler.Constraints.new_constraint(
                f"z_{self.K, j2} beta_{j2} >= z_{self.K,j1} - (1 - beta_{j2}) U_{j1}",
                label="same_for_data",
            ):
                continue
            for i in range(self.n[self.K - 1]):
                if (self.K - 1, i) in self.stable_inactives_neurons:
                    continue
                # elif (self.K - 1, i) in self.stable_actives_neurons:
                #     # beta_j2 * z_j2 - W part
                #     self.handler.Constraints.add_quad_variable_bounding(
                #         layer=self.K - 1,
                #         neuron=i,
                #         class_label=j2,
                #         value=self.W[self.K - 1][j2][i],
                #         alpha_1=self.alpha_1,
                #         alpha_2=self.alpha_2,
                #         type="upper",
                #     )
                else:
                    # beta_j2 * z_j2 - W part
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=self.K - 1,
                        neuron1=i,
                        var2="beta",
                        class_label=j2,
                        value=self.W[self.K - 1][j2][i],
                        front_of_matrix1=False,
                    )

                # - z_j1 - W part
                self.handler.Constraints.add_linear_variable(
                    var="z",
                    layer=self.K - 1,
                    neuron=i,
                    value=-self.W[self.K - 1][j1][i],
                    front_of_matrix=False,
                )
            # beta_j2 * z_j2 - b part
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=self.b[self.K - 1][j2],
            )
            # - z_j1 - b part
            self.handler.Constraints.add_constant(value=-self.b[self.K - 1][j1])

            # - beta_j2 * U_j1
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=-self.handler.Constraints.U_above_zero[self.K][j1],
            )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=-self.handler.Constraints.U_above_zero[self.K][j1],
            )


def z_j2_beta_j2_less_than_zj(self):
    """
    Add the constraint z_2 beta_2 <= (1 - beta_1) z_1 + beta_2 U_2 - (1 - beta_1) L_1 + beta_2 (L_1 - z_1)   (12)
    ie      z_2 beta_2 + beta_1 z_1 - z_1 + beta_2 z_1 <=  beta_2 U_2 + beta_1 L_1 - L_1 + beta_2 L_1
    """
    assert self.BETAS

    for j2 in self.ytargets:
        if j2 == self.ytrue:
            continue
        for j1 in self.ytargets:
            if j1 == self.ytrue or j1 == j2:
                continue
            if self.handler.Constraints.new_constraint(
                f"z_{j2} beta_{j2} <= (1 - beta_{j1}) z_{j1} + beta_{j2} U_{j2} - (1 - beta_{j1}) L_{j1} + beta_{j2} (L_{j1} - z_{j1})",
                label="same_for_data",
            ):
                continue

            for i in range(self.n[self.K - 1]):
                if (self.K - 1, i) in self.stable_inactives_neurons:
                    continue

                # if (self.K - 1, i) in self.stable_actives_neurons:
                #     # beta_j2 * z_j2 - W part
                #     self.handler.Constraints.add_quad_variable_bounding(
                #         layer=self.K - 1,
                #         neuron=i,
                #         class_label=j2,
                #         value=self.W[self.K - 1][j2][i],
                #         type="lower",
                #         alpha_1=self.alpha_1,
                #         alpha_2=self.alpha_2,
                #     )
                #     # beta_j1 * z_j1 - W part
                #     self.handler.Constraints.add_quad_variable_bounding(
                #         layer=self.K - 1,
                #         neuron=i,
                #         class_label=j1,
                #         value=self.W[self.K - 1][j1][i],
                #         type="lower",
                #         alpha_1=self.alpha_1,
                #         alpha_2=self.alpha_2,
                #     )
                #     # beta_j2 * z_j1 - W part
                #     self.handler.Constraints.add_quad_variable_bounding(
                #         layer=self.K - 1,
                #         neuron=i,
                #         class_label=j2,
                #         value=self.W[self.K - 1][j1][i],
                #         type="lower",
                #         alpha_1=self.alpha_1,
                #         alpha_2=self.alpha_2,
                #     )
                else:
                    # beta_j2 * z_j2 - W part
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=self.K - 1,
                        neuron1=i,
                        var2="beta",
                        class_label=j2,
                        value=self.W[self.K - 1][j2][i],
                        front_of_matrix1=False,
                    )
                    # beta_j1 * z_j1 - W part
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=self.K - 1,
                        neuron1=i,
                        var2="beta",
                        class_label=j1,
                        value=self.W[self.K - 1][j1][i],
                        front_of_matrix1=False,
                    )
                    # beta_j2 * z_j1 - W part
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=self.K - 1,
                        neuron1=i,
                        var2="beta",
                        class_label=j2,
                        value=self.W[self.K - 1][j1][i],
                        front_of_matrix1=False,
                    )
                # - z_j1 - W part
                self.handler.Constraints.add_linear_variable(
                    var="z",
                    layer=self.K - 1,
                    neuron=i,
                    value=-self.W[self.K - 1][j1][i],
                    front_of_matrix=False,
                )

            # beta_j2 * z_j2 - b part
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=self.b[self.K - 1][j2],
            )

            # - z_j1 - b part
            self.handler.Constraints.add_constant(value=-self.b[self.K - 1][j1])

            # beta_j1 * z_j1 - b part
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j1,
                value=self.b[self.K - 1][j1],
            )

            # beta_j2 * z_j1 - b part
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=self.b[self.K - 1][j1],
            )

            # - beta_j2 * U_j2
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=-self.handler.Constraints.U_above_zero[self.K][j2],
            )

            # - beta_j1 * L_j1
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j1,
                value=-self.handler.Constraints.L[self.K][j1],
            )

            # - beta_j2 * L_j1
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=-self.handler.Constraints.L[self.K][j1],
            )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.up,
                bound=-self.handler.Constraints.L[self.K][j1],
            )


# *****************************************************************************************************************
# *********************************************ZBAR ***************************************************************
# *****************************************************************************************************************
def zbar_sum_beta_z(self):
    """
    Add the constraint zbar = sum(betaj * zj) for the beta variable (ensuring beta is in [0,1])
    """
    assert self.ZBAR
    assert self.BETAS

    name_cstr = "zbar = sum("
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        name_cstr += f"beta_{j} * z_{self.K-1,j}"
        if j < len(self.ytargets) - 1:
            name_cstr += " + "
    name_cstr += ")"

    if self.handler.Constraints.new_constraint(name_cstr, label = "same_for_data"):
        return
    self.handler.Constraints.add_linear_variable(
        var="zbar",
        value=1,
    )
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons or (
                self.K - 1,
                i,
            ) in self.stable_actives_neurons:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=self.K - 1,
                neuron2=i,
                value=-self.W[self.K - 1][j][i],
            )

        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=-self.b[self.K - 1][j],
        )

    self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)


def zbar_max_z(self):
    """
    Add the constraint zbar >= max(zj)
    """
    assert self.ZBAR
    assert self.BETAS

    for j in self.ytargets:
        if j == self.ytrue:
            continue
        # zbar >= zj
        if self.handler.Constraints.new_constraint(f"zbar >= z_{self.K,j}", label="same_for_data"):
            continue
        self.handler.Constraints.add_linear_variable(
            var="zbar",
            value=1,
        )
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons or (
                self.K - 1,
                i,
            ) in self.stable_actives_neurons:
                continue
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K - 1,
                neuron=i,
                value=-self.W[self.K - 1][j][i],
            )

        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.lo, bound=self.b[self.K - 1][j]
        )
