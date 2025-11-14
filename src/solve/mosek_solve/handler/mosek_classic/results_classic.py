from typing import List
import mosek
import logging
import numpy as np

from tools import get_project_path
from ...get_variables import print_solution_to_file_for_cb_solver


logger_mosek = logging.getLogger("Mosek_logger")


def add_all_infos_optimal_values_to_dic(self, cuts: List):
    """
    Add all the information about the optimal values found to the dictionnary for the benchmark.
    """
    self.primal_obj_value = self.task.getprimalobj(mosek.soltype.itr)
    logger_mosek.debug(
        "Optimal solution found with objective value: %s", self.primal_obj_value
    )    
    self.dual_obj_value = self.task.getdualobj(mosek.soltype.itr)
    if self.verbose : 
        print("Optimal primal solution found with objective value: ", self.primal_obj_value)
        print("Optimal dual objective value: ", self.dual_obj_value)
    logger_mosek.info("Dual objective value: %s", self.dual_obj_value)
    if self.verbose :
        print(f"CALLBACK : constant = {self.Objective.constant}")
    self.optimal_value = self.primal_obj_value + self.Objective.constant
    print(
        f"CALLBACK : Optimal value (with added constant): {self.optimal_value} : with cuts {cuts}"
    )
    self.is_robust = self.optimal_value >= 0
    if self.verbose :
        print("Is robust: ", self.is_robust)
    self.compute_solutions(cuts, print_sol = True)
    dic_sol = {"optimal_value": self.optimal_value}
    dic_sol.update({"primal_obj_value": self.primal_obj_value})
    dic_sol.update({"dual_obj_value": self.dual_obj_value})
    return dic_sol


@staticmethod
def reconstruct_matrix(size, tab_triang):
    """
            Reconstruct the symmetric matrix
    from the values of the lower triangular part given in a one-dimensional array.

    Args:
    size (int): dimension of the square matrix
    tab_triang (list): array of values from the lower triangular part of the matrix
    """
    mat = np.zeros((size, size))
    tri_indices = np.triu_indices(size)
    mat[tri_indices] = tab_triang
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def is_status_optimal(self):
    """
    Check if the status of the solver is optimal.

    Returns
    -------
    bool
        True if the status is optimal, False otherwise.
    """
    return self.status == mosek.solsta.optimal


def is_status_infeasible(self):
    """
    Check if the status of the solver is infeasible.
    Returns
    -------
    bool
        True if the status is infeasible, False otherwise.
    """
    return (
        self.status == mosek.solsta.dual_infeas_cer
        or self.status == mosek.solsta.prim_infeas_cer
    )


def is_status_unknown(self):
    """
    Check if the status of the solver is unknown.
    Returns
    -------
    bool
        True if the status is unknown, False otherwise.
    """
    return self.status == mosek.solsta.unknown
