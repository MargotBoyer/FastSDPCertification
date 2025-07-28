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
from .certification_problem_objective import objective_Mzbar
from .certification_problem_constraints_bounds import (
    quad_bounds,
    McCormick_inter_layers,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
)
from .certification_problem_constraints_relu import (
    ReLU_constraint_Lan,
    ReLU_triangularization,
)
from .certification_problem_constraints_beta import (
    discrete_betas,
    sum_betas_equals_1,
    McCormick_beta_z,
    McCormick_beta_z_with_penultimate_layer,
    betai_betaj,
    zbar_sum_beta_z,
    zbar_max_z,
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
    objective_Mzbar,
    ReLU_constraint_Lan,
    quad_bounds,
    discrete_betas,
    sum_betas_equals_1,
    zbar_sum_beta_z,
    betai_betaj,
    zbar_max_z,
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
)
class MzbarSDP(MosekSolver):
    def __init__(self, **kwargs):
        super().__init__(
            certification_model_name="MzbarSDP",
            BETAS=True,
            BETAS_Z=True,
            ZBAR=True,
            **kwargs,
        )
        print("MZBAR MATRIX BY LAYERS", self.MATRIX_BY_LAYERS)

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Mzbar()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """
        # RELU
        self.ReLU_constraint_Lan()
        if "triangularization" in cuts:
            self.ReLU_triangularization()

        # BOUNDS
        self.quad_bounds()

        # BETA
        self.discrete_betas()
        self.sum_betas_equals_1()
        self.betai_betaj()

        # ZBAR
        self.zbar_sum_beta_z()
        self.zbar_max_z()

        # Tij
        if "Tij" in cuts:
            self.McCormick_beta_z(layer=self.K - 1)

        if "Tij_before_penultimate_layer" in cuts:
            self.McCormick_beta_z(layer=self.K - 2)

        # McCormick on beta z : with z logits
        if "McC_betaz_logits" in cuts:
            self.McCormick_beta_z_with_penultimate_layer()

        # Some cuts comparing different logits
        if "logits_comparaison_11" in cuts:
            self.z_j2_beta_j2_greater_than_zj()

        if "logits_comparaison_12" in cuts:
            self.z_j2_beta_j2_less_than_zj()
        # RLT
        if "RLT" in cuts:
            self.add_RLT_constraints(p=self.RLT_prop)

        # McCormick
        if "allMC" in cuts:
            self.all_Mc_Cormick_all_layers()

        # MATRIX BY LAYERS
        if self.MATRIX_BY_LAYERS:
            self.matrix_by_layers_rec(only_linear_constraints=True)

        self.first_term_equal_zero()

        self.handler.Constraints.end_constraints()
