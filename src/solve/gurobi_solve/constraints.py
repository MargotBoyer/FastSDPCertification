import gurobipy as gp
from gurobipy import GRB
import logging

logger_gurobi = logging.getLogger("Gurobi_logger")


def quad_bounds(self):
    """Add quadratic bounds on all layers."""
    for k in range((self.K + 1 if self.LAST_LAYER else self.K)):
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                continue
            self.m.addConstr(
                self.z[k, j] * self.z[k, j]
                - (self.L[k][j] + self.U_above_zero[k][j]) * self.z[k, j]
                + self.L[k][j] * self.U_above_zero[k][j]
                <= 0
            )


def ReLU_constraint_Lan(self):
    """ReLU quadratic exact constraint on continuous variables"""
    for k in range(1, self.K):
        for j in range(self.n[k]):
            self.m.addConstr(
                gp.quicksum(
                    self.W[k - 1][j][i] * self.z[k - 1, i] for i in range(self.n[k - 1])
                )
                + self.b[k - 1][j]
                <= self.z[k, j]
            )
            self.m.addConstr(
                (
                    self.z[k, j]
                    - gp.quicksum(
                        self.W[k - 1][j][i] * self.z[k - 1, i]
                        for i in range(self.n[k - 1])
                    )
                    - self.b[k - 1][j]
                )
                * self.z[k, j]
                == 0
            )


def RELU_triangular_constraint(self):
    """ReLU quadratic exact constraint on continuous variables"""
    for k in range(1, self.K):
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons :
                continue
            
            #print(f"Adding triangular ReLU constraint for layer {k}, neuron {j}")
            self.m.addConstr(
                gp.quicksum(
                    self.W[k - 1][j][i] * self.z[k - 1, i] for i in range(self.n[k - 1]) if (k - 1, i) not in self.stable_inactives_neurons
                )
                + self.b[k - 1][j]
                <= self.z[k, j]
            )
            if abs(self.U[k][j] - self.L[k][j]) <= 1e-6:
                logger_gurobi.warning(
                    f"Layer {k}, Neuron {j} : L={self.L[k][j]} and U={self.U[k][j]} are equal, triangular ReLU constraint is not added."
                )
                continue

            rel_u = max(self.U[k][j], 0)
            rel_l = max(self.L[k][j], 0)
            k_cst = (rel_u - rel_l) / (self.U[k][j] - self.L[k][j])
            B_k_j = k_cst * (self.b[k - 1][j] - self.L[k][j]) + rel_l

            self.m.addConstr(
                self.z[k, j]
                <= gp.quicksum(
                    k_cst * self.W[k - 1][j][i] * self.z[k - 1, i] 
                    for i in range(self.n[k - 1]) if (k - 1, i) not in self.stable_inactives_neurons
                )
                + B_k_j
            )


def sum_beta_equals_1(self):
    """Sum of beta variables equals 1"""
    self.m.addConstr(
        gp.quicksum(self.beta[j] for j in self.ytargets if j != self.ytrue) == 1
    )


def zbar_sum_betaz(self):
    """zbar is equal to the sum of beta * z"""
    if self.LAST_LAYER:
        raise NotImplementedError(
            "zbar_sum_betaz is not implemented for the last layer."
        )
    else:
        self.m.addConstr(
            gp.quicksum(
                self.beta[j]
                * (
                    gp.quicksum(
                        self.z[self.K - 1, i] * self.W[self.K - 1][j][i]
                        for i in range(self.n[self.K - 1])
                    )
                    + self.b[self.K - 1][j]
                )
                for j in self.ytargets
                if j != self.ytrue
            )
            == self.zbar
        )
