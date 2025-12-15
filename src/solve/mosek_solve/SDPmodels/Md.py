import numpy as np
import mosek
import yaml
import os
import sys
import logging
from typing import List


from tools.utils import infinity, add_functions_to_class


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from ..mosek_generic_solver import MosekSolver
from networks import ReLUNN
from .certification_problem_objective import objective_Md
from .certification_problem_constraints_bounds import (
    L2_ball_bounds,
    quad_bounds,
    McCormick_inter_layers,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
)
from .certification_problem_constraints_relu import (
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    ReLU_triangularization,
)
from .certification_problem_constraints_beta import (
    discrete_betas,
    sum_betas_equals_1,
    McCormick_beta_z,
    McCormick_beta_z_with_penultimate_layer,
    betai_betaj,
    z_j2_beta_j2_greater_than_zj,
    z_j2_beta_j2_less_than_zj,
)
from .certification_problem_constraints_division_by_layers import (
    matrix_by_layers_rec,
)
from .certification_problem_constraints_rlt import add_RLT_constraints
from .certification_problem_constraints_sdp import first_term_equal_zero


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    objective_Md,
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    quad_bounds,
    discrete_betas,
    sum_betas_equals_1,
    betai_betaj,
    ReLU_triangularization,
    matrix_by_layers_rec,
    add_RLT_constraints,
    McCormick_inter_layers,
    first_term_equal_zero,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
    McCormick_beta_z,
    McCormick_beta_z_with_penultimate_layer,
    z_j2_beta_j2_greater_than_zj,
    z_j2_beta_j2_less_than_zj,
    L2_ball_bounds
)
class MdSDP(MosekSolver):
    def __init__(self, **kwargs):

        super().__init__(
            certification_model_name="MdSDP", BETAS=True, BETAS_Z=True, **kwargs
        )

        logger_mosek.debug(f"Bounds for the network :  {self.L} and {self.U}")
        print("ytargets in MdSDP:", self.ytargets)

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Md()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """
        # RELU
        ub_neurons = "ReLU_active_ub_neurons" in cuts
        lb_neurons = "ReLU_active_lb_neurons" in cuts
        ub_decomposed = "ReLU_active_ub_decomposed" in cuts
        lb_decomposed = "ReLU_active_lb_decomposed" in cuts

        print("STUDY : Adding ReLU constraints...")
        self.ReLU_constraint_Lan()
        print("STUDY : ReLU constraints added.")

        if "triangularization" in cuts:
            self.ReLU_triangularization()

        # BOUNDS
        self.quad_bounds()
        if self.norm == "L2":
            self.L2_ball_bounds()

        # BETA
        self.discrete_betas()
        self.sum_betas_equals_1()
        self.betai_betaj()

        # Tij
        if "Tij" in cuts:
            self.McCormick_beta_z(layer=self.K - 1)

        if "Tij_before_penultimate_layer" in cuts:
            self.McCormick_beta_z(layer=self.K - 2)

        # McCormick on beta z : with z logits
        if "McC_betaz_logits" in cuts:
            self.McCormick_beta_z_with_penultimate_layer()

        # # Some cuts comparing different logits
        if "beta_logits_comparaison_2" in cuts:
            self.z_j2_beta_j2_greater_than_zj()

        if "beta_logits_comparaison_1" in cuts:
            self.z_j2_beta_j2_less_than_zj()

        # RLT
        if "RLT" in cuts:
            self.add_RLT_constraints(p=self.RLT_prop)

        # McCormick
        if "allMC" in cuts:
            self.all_Mc_Cormick_all_layers()

        self.first_term_equal_zero()

        # MATRIX BY LAYERS
        if self.MATRIX_BY_LAYERS:
            self.matrix_by_layers_rec(only_linear_constraints=True)

        self.handler.Constraints.end_constraints()
