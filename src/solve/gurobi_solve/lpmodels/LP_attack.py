import logging
import numpy as np

from ..gurobi_generic_solver import GurobiSolver
from ..objective import add_objective_Lan
from ..constraints import RELU_triangular_constraint, quad_bounds
from ..variables import add_variable_z, _add_variable_z
from tools.utils import add_functions_to_class
from ..callback import NonConvexQuadraticProgramCallback

logger_gurobi = logging.getLogger("Gurobi_logger")


@add_functions_to_class(
    add_objective_Lan,
    RELU_triangular_constraint,
    quad_bounds,
    add_variable_z,
    _add_variable_z,
)
class ClassicLP(GurobiSolver):
    """
    A solver that uses Gurobi to solve the optimization problem.
    """

    def __init__(
        self,
        **kwargs,
    ):
        #print("kwargs in init LP : ", kwargs)

        super().__init__(certification_model_name="LP", **kwargs)


        # self.possible_targets = [
        #     target for target in self.ytargets if target != self.ytrue
        # ]
        # if "ytarget" in kwargs:
        #     self.ytarget = kwargs["ytarget"]
        # else:
        #     self.ytarget = np.random.choice(self.possible_targets)

        print("LP attack initialized with ytarget:", self.ytarget)

    def add_objective(self):
        self.add_objective_Lan()

    def add_variables(self):
        """
        Add variables to the model.
        """
        self.add_variable_z()

    def add_constraints(self):
        """
        Add constraints to the model.
        """
        # RELUMIX
        self.RELU_triangular_constraint()

        # BOUNDS
        # self.quad_bounds()
