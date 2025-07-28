import logging
import numpy as np

from ..gurobi_generic_solver import GurobiSolver
from ..objective import add_objective_Md
from ..constraints import ReLU_constraint_Lan, quad_bounds, sum_beta_equals_1
from ..variables import (
    add_variable_z,
    _add_variable_z,
    add_variable_beta,
    _add_variable_beta,
)
from ..callback import NonConvexQuadraticProgramCallback
from tools.utils import add_functions_to_class

logger_gurobi = logging.getLogger("Gurobi_logger")


@add_functions_to_class(
    add_objective_Md,
    ReLU_constraint_Lan,
    quad_bounds,
    sum_beta_equals_1,
    add_variable_z,
    _add_variable_z,
    add_variable_beta,
    _add_variable_beta,
)
class MdQuad(GurobiSolver):
    """
    A solver that uses Gurobi to solve the optimization problem.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(certification_model_name="MdQuad", BETAS=True, **kwargs)


    def add_objective(self):
        self.add_objective_Md()

    def add_variables(self):
        """
        Add variables to the model.
        """
        self.add_variable_z()
        self.add_variable_beta()

    def add_constraints(self):
        """
        Add constraints to the model.
        """
        # RELUMIX
        self.ReLU_constraint_Lan()

        # BOUNDS
        # self.quad_bounds()

        # BETAS
        self.sum_beta_equals_1()
