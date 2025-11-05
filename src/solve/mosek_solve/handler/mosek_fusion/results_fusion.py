from typing import List
import mosek
import logging

from tools import get_project_path


logger_mosek = logging.getLogger("Mosek_logger")


def add_all_infos_optimal_values_to_dic(self, cuts: List, verbose: bool = False):
    """
    Add all the information about the optimal values found to the dictionnary for the benchmark.
    """
    self.primal_obj_value = self.model.primalObjValue()
    logger_mosek.debug(
        "Optimal solution found with objective value: %s", self.primal_obj_value
    )
    print("Optimal solution found with objective value: ", self.primal_obj_value)
    self.dual_obj_value = self.model.dualObjValue()
    logger_mosek.info("Dual objective value: %s", self.dual_obj_value)
    self.optimal_value = self.primal_obj_value  # Pas de constante hors du model ici
    print(
        f"Optimal value (no added constant, already written in model): {self.optimal_value} : with cuts {cuts}"
    )
    # self.compute_solutions(cuts, verbose)
    self.is_robust = self.optimal_value >= 0
    dic_sol = {"optimal_value": self.optimal_value}
    dic_sol.update({"primal_obj_value": self.primal_obj_value})
    dic_sol.update({"dual_obj_value": self.dual_obj_value})
    return dic_sol


def is_status_optimal(self):
    """
    Check if the status of the solver is optimal.

    Returns
    -------
    bool
        True if the status is optimal, False otherwise.
    """
    return self.model.getPrimalSolutionStatus() == mosek.fusion.SolutionStatus.Optimal


def is_status_infeasible(self):
    """
    Check if the status of the solver is infeasible.

    Returns
    -------
    bool
        True if the status is infeasible, False otherwise.
    """
    return (
        self.model.getProblemStatus() == mosek.fusion.ProblemStatus.PrimalInfeasible
        or self.model.getProblemStatus() == mosek.fusion.ProblemStatus.DualInfeasible
    )


def is_status_unknown(self):
    """
    Check if the status of the solver is unknown.

    Returns
    -------
    bool
        True if the status is unknown, False otherwise.
    """
    return self.model.getPrimalSolutionStatus() == mosek.fusion.SolutionStatus.Unknown
