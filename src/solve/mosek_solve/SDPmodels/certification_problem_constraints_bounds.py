import mosek
from tools.utils import infinity
import logging

logger_mosek = logging.getLogger("Mosek_logger")


# ********************************************* BOUNDS ***************************************************************
def quad_bounds(self):
    print("Adding quadratic bounds constraint")

    for k in range(self.K + 1 if self.LAST_LAYER else self.K):
        print(f"Adding quadratic bounds constraint for layer {k}")
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons or (
                k,
                j,
            ):
                continue
            elif (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                continue
            # zk² - (U+L) + UL <= 0
            if not self.use_inactive_neurons:
                assert self.handler.Constraints.U[k][j] >= 0
            if self.handler.Constraints.new_constraint(
                f"z_{k,j}^2 - (U+L) z_{k,j} + UL <= 0", label = "same_for_data"
            ):
                continue
            front_of_matrix = (
                True
                if (
                    (k < self.K - 1 and not self.LAST_LAYER)
                    or (k < self.K and self.LAST_LAYER)
                )
                else False
            )
            self.handler.Constraints.add_quad_variable(
                var1="z",
                layer1=k,
                neuron1=j,
                var2="z",
                layer2=k,
                neuron2=j,
                value=1,
                front_of_matrix1=front_of_matrix,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=k,
                neuron=j,
                value=-(
                    self.handler.Constraints.U_above_zero[k][j]
                    + self.handler.Constraints.L[k][j]
                ),
                front_of_matrix=front_of_matrix,
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.up,
                bound=-(
                    self.handler.Constraints.U_above_zero[k][j]
                    * self.handler.Constraints.L[k][j]
                ),
            )
    if self.ZBAR:
        if self.handler.Constraints.new_constraint(
            f"zbar - (max(U_{self.K}+ max(L_{self.K})) zbar + (max(U_{self.K} * max(L_{self.K})) <= 0"
        ):
            return
        self.handler.Constraints.add_quad_variable(
            var1="zbar",
            var2="zbar",
            value=1,
        )
        self.handler.Constraints.add_linear_variable(
            var="zbar",
            value=-(
                max(self.handler.Constraints.U[self.K])
                + min(self.handler.Constraints.L[self.K])
            ),
        )
        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.up,
            bound=-(
                max(self.handler.Constraints.U[self.K])
                * min(self.handler.Constraints.L[self.K])
            ),
        )


# ********************************************* McCormick ***************************************************************


def McCormick_inter_layers(self, k: int, neuron_prev: int, neuron_next : int):
    """
    Add 3 McCormick constraints for the inter-layer connections from the paper Lan.
    """
    # *************** Constraint (12b) in Lan **********************

    if self.handler.Constraints.L[k - 1][neuron_prev] < 0:
        lb_prev = 0
    else :
        lb_prev = self.handler.Constraints.L[k - 1][neuron_prev]
    if self.handler.Constraints.L[k][neuron_next] < 0:
        lb_next = 0
    else :  
        lb_next = self.handler.Constraints.L[k][neuron_next]
    # if k > 0 : 
    #     print(f"STUDY MCCORMICK - L_{k-1} {neuron_prev}={self.handler.Constraints.L[k-1][neuron_prev]} L_{k} {neuron_next}={self.handler.Constraints.L[k][neuron_next]}")
    # z_{k+1j} z_{ki} >= z_{k+1j} * L_{ki} + z_{ki} * L_{k+1j} - L_{ki} * L_{k+1j}
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {k - 1}, neuron {neuron_prev}    ; Layer {k - 1+1}, neuron {neuron_next}  - 12b (RLT)"
    ):
        return

    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=k,
        neuron1=neuron_next,
        var2="z",
        layer2=k - 1,
        neuron2=neuron_prev,
        value=1,
        front_of_matrix1=False,
        front_of_matrix2=True,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=k,
        neuron=neuron_next,
        value=-lb_prev,
        front_of_matrix=False,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=k - 1,
        neuron=neuron_prev,
        value=-lb_next,
        front_of_matrix=True,
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.lo,
        bound=-lb_prev
        * lb_next,
    )

    # *************** Constraint (12c) in Lan **********************
    # z_{k - 1+1j} z_{k - 1i} <= z_{k - 1+1j} * U_{k - 1j} + z_{k - 1i} * L_{k - 1+1j} - U_{k - 1j} * L_{k - 1+1j}
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {k - 1}, neuron {neuron_prev}    ; Layer {k - 1+1}, neuron {neuron_next}  - 12c (RLT)"
    ):
        return
    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=k,
        neuron1=neuron_next,
        var2="z",
        layer2=k - 1,
        neuron2=neuron_prev,
        value=1,
        front_of_matrix1=False,
        front_of_matrix2=True,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=k,
        neuron=neuron_next,
        value=-self.handler.Constraints.U_above_zero[k - 1][neuron_prev],
        front_of_matrix=False,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=k - 1,
        neuron=neuron_prev,
        value=-lb_next,
        front_of_matrix=True,
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.up,
        bound=-self.handler.Constraints.U_above_zero[k - 1][neuron_prev]
        * lb_next,
    )

    # z_{k - 1+1j} z_{k - 1i} <= z_{k - 1+1j} * L_{k - 1j} + z_{k - 1i} * U_{k - 1+1j} - L_{k - 1j} * U_{k - 1+1j}
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {k - 1}, neuron {neuron_prev}    ; Layer {k - 1+1}, neuron {neuron_next}  - 12c second part (RLT)"
    ):
        return
    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=k,
        neuron1=neuron_next,
        var2="z",
        layer2=k - 1,
        neuron2=neuron_prev,
        value=1,
        front_of_matrix1=False,
        front_of_matrix2=True,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=k,
        neuron=neuron_next,
        value=-lb_prev,
        front_of_matrix=False,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=k - 1,
        neuron=neuron_prev,
        value=-self.handler.Constraints.U_above_zero[k][neuron_next],
        front_of_matrix=True,
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.up,
        bound=-lb_prev
        * self.handler.Constraints.U_above_zero[k][neuron_next],
    )


# ***********************************************************************************************************************
# ***********************************************************************************************************************
# ********************************************* McCormick ***************************************************************
# ***********************************************************************************************************************
# ***********************************************************************************************************************
def is_front_of_matrix(self, layer1: int, layer2: int):
    """
    return front_of_matrix for layer1 and layer2
    """
    if layer1 == layer2:
        if layer1 == 0:
            return True, True
        elif layer1 == (self.K if self.LAST_LAYER else self.K - 1):
            return False, False
        else:
            return True, True
    elif layer2 < layer1:
        return False, True
    else:
        raise ValueError(f"Layer {layer1} < Layer {layer2}")


def all_4_McCormick(self, layer1: int, neuron1: int, layer2: int, neuron2: int):
    """
    Add all 4 McCormick constraints for all layers.
    """
    if layer1 == layer2 and neuron1 == neuron2:
        logger_mosek.error(
            f"McCormick constraints for the same neuron {layer1} {neuron1} and {layer2} {neuron2}"
        )
    if layer1 < layer2:
        layer1, layer2, neuron1, neuron2 = layer2, layer1, neuron2, neuron1

    front_of_matrix_layer1, front_of_matrix_layer2 = self.is_front_of_matrix(
        layer1, layer2
    )

    # z_{k1 j1} z_{k2 j2} >= z_{k1 j1} * L_{k2 j2} + z_{k2 j2} * L_{k+1j} - L_{k1 j1} * L_{k2 j2}
    name_cstr = f"z_({layer1} {neuron1}) z_({layer2} {neuron2}) >= z_({layer1} {neuron1}) * L_({layer2} {neuron2}) + z_({layer2} {neuron2}) * L_({layer1} {neuron1}) - L_({layer1} {neuron1}) * L_({layer2} {neuron2})"
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {layer1}, neuron {neuron1}    ; Layer {layer2}, neuron {neuron2} - {name_cstr}"
    ):
        return

    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=1,
        front_of_matrix1=front_of_matrix_layer1,
        front_of_matrix2=front_of_matrix_layer2,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer1,
        neuron=neuron1,
        value=-self.handler.Constraints.L[layer2][neuron2],
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer2,
        neuron=neuron2,
        value=-self.handler.Constraints.L[layer1][neuron1],
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.lo,
        bound=-self.handler.Constraints.L[layer1][neuron1]
        * self.handler.Constraints.L[layer2][neuron2],
    )

    # z_{k1 j1} z_{k2 j2} <= z_{k1 j1} * U_{k2 j2} + z_{k2 j2} * L_{k1 j1} - U_{k2 - j2} * L_{k1 j1}
    name_cstr = f"z_({layer1} {neuron1}) z_({layer2} {neuron2}) >= z_({layer1} {neuron1}) * U_({layer2} {neuron2}) + z_({layer2} {neuron2}) * L_({layer1} {neuron1}) - L_({layer1} {neuron1}) * U_({layer2} {neuron2})"
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {layer1}, neuron {neuron1}    ; Layer {layer2}, neuron {neuron2} - {name_cstr}"
    ):
        return
    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=1,
        front_of_matrix1=front_of_matrix_layer1,
        front_of_matrix2=front_of_matrix_layer2,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer1,
        neuron=neuron1,
        value=-self.handler.Constraints.U_above_zero[layer2][neuron2],
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer2,
        neuron=neuron2,
        value=-self.handler.Constraints.L[layer1][neuron1],
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.up,
        bound=-self.handler.Constraints.U_above_zero[layer2][neuron2]
        * self.handler.Constraints.L[layer1][neuron1],
    )
    # z_{k - 1+1j} z_{k - 1i} <= z_{k - 1+1j} * L_{k - 1j} + z_{k - 1i} * U_{k - 1+1j} - L_{k - 1j} * U_{k - 1+1j}
    name_cstr = f"z_({layer1} {neuron1}) z_({layer2} {neuron2}) >= z_({layer1} {neuron1}) * L_({layer2} {neuron2}) + z_({layer2} {neuron2}) * U_({layer1} {neuron1}) - U_({layer1} {neuron1}) * L_({layer2} {neuron2})"
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {layer1}, neuron {neuron1}    ; Layer {layer2}, neuron {neuron2}  - {name_cstr}"
    ):
        return
    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=1,
        front_of_matrix1=front_of_matrix_layer1,
        front_of_matrix2=front_of_matrix_layer2,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer1,
        neuron=neuron1,
        value=-self.handler.Constraints.L[layer2][neuron2],
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer2,
        neuron=neuron2,
        value=-self.handler.Constraints.U_above_zero[layer1][neuron1],
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.up,
        bound=-self.handler.Constraints.L[layer2][neuron2]
        * self.handler.Constraints.U_above_zero[layer1][neuron1],
    )

    # z_{k1 j1} z_{k2 j2} >= z_{k1 j1} * U_{k2 j2} + z_{k2 j2} * U_{k+1j} - U_{k1 j1} * U_{k2 j2}
    name_cstr = f"z_({layer1} {neuron1}) z_({layer2} {neuron2}) >= z_({layer1} {neuron1}) * U_({layer2} {neuron2}) + z_({layer2} {neuron2}) * U_({layer1} {neuron1}) - U_({layer1} {neuron1}) * U_({layer2} {neuron2})"
    if self.handler.Constraints.new_constraint(
        f"McCormick - Layer {layer1}, neuron {neuron1}    ; Layer {layer2}, neuron {neuron2} - {name_cstr}"
    ):
        return

    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=1,
        front_of_matrix1=front_of_matrix_layer1,
        front_of_matrix2=front_of_matrix_layer2,
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer1,
        neuron=neuron1,
        value=-self.handler.Constraints.U_above_zero[layer2][neuron2],
    )
    self.handler.Constraints.add_linear_variable(
        var="z",
        layer=layer2,
        neuron=neuron2,
        value=-self.handler.Constraints.U_above_zero[layer1][neuron1],
    )
    self.handler.Constraints.add_bound(
        bound_type=mosek.boundkey.lo,
        bound=-self.handler.Constraints.U_above_zero[layer1][neuron1]
        * self.handler.Constraints.U_above_zero[layer2][neuron2],
    )


def all_Mc_Cormick_all_layers(self):
    for k in range(self.K + 1 if self.LAST_LAYER else self.K):
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                continue
            elif (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                continue
            if k == 0:
                k2_list = [0]
            else:
                if self.MATRIX_BY_LAYERS:
                    k2_list = [k - 1, k]
                else:
                    k2_list = range(k, self.K + 1 if self.LAST_LAYER else self.K)
            for k2 in k2_list:
                for j2 in range(j if k == k2 else self.n[k2]):
                    if (k2, j2) in self.stable_inactives_neurons or (
                        k2,
                        j2,
                    ) in self.stable_actives_neurons:
                        continue
                    self.all_4_McCormick(k, j, k2, j2)
