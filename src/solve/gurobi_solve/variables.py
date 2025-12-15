import gurobipy as gp
from gurobipy import GRB
import logging

logger_gurobi = logging.getLogger("Gurobi_logger")


def _add_variable_z(
    self,
    layer: int,
    neuron: int,
    lb: float,
    ub: float,
    delta: float = 0.0000001,
):
    """
    Add the variable z to the model.
    """
    self.z[layer, neuron] = self.m.addVar(lb=lb - delta, ub=ub+delta, vtype=GRB.CONTINUOUS)
    logger_gurobi.debug(
        "Adding variable z[%s,%s] with bounds [%s, %s]",
        layer,
        neuron,
        lb - delta,
        ub + delta,
    )


def add_variable_z(
    self,
    impose_positive: bool = True,
    delta: float = 0.000001,
):
    """
    Add the variable z to the model.
    """
    self.z = gp.tupledict()
    max_layer_z = self.K + 1 if self.LAST_LAYER else self.K

    for layer in range(max_layer_z):
        lb_ = self.L[layer]
        ub_ = self.U[layer]
        if impose_positive and (layer > 0) and (layer < self.K):
            lb_ = [0] * self.n[layer]
        for neuron in range(self.n[layer]):
            if (layer, neuron) in self.stable_inactives_neurons:
                continue
            #print(f"Adding variable z[{layer},{neuron}] with bounds [{lb_[neuron]}, {ub_[neuron]}]")
            self._add_variable_z(
                layer,
                neuron,
                lb_[neuron],
                max(ub_[neuron], 0),
                delta=delta,
            )
    return self.z


def _add_variable_beta(self, class_label: int, relax: bool = False):
    """
    Add the variable beta to the model.
    """
    if relax:
        self.beta[class_label] = self.m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)
    else:
        self.beta[class_label] = self.m.addVar(vtype=GRB.BINARY)


def add_variable_beta(
    self,
    relax: bool = False,
):
    """
    Add the variable beta to the model.
    """
    self.beta = gp.tupledict()
    for class_label in self.ytargets:
        if class_label == self.ytrue:
            continue
        self._add_variable_beta(class_label, relax=relax)
    return self.beta


def add_variable_zbar(self):
    """
    Add the variable z to the model.
    """
    self.zbar = self.m.addVar(
        lb=min(self.L[self.K]),
        ub=max(self.U[self.K]),
        vtype=GRB.CONTINUOUS,
        name="zbar",
    )
    return self.zbar
