from typing import List
import mosek
import numpy as np
from mosek.fusion import Expr, Domain, Matrix, Model, Variable
import mosek.fusion.pythonic

from ..indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from ..indexes_variables import Indexes_Variables_for_Mosek_Solver

from ..constraints import CommonConstraints
import os
import sys
import logging
from tools import (
    infinity,
    sort_lists_by_first,
    deduplicate_and_sum,
    add_functions_to_class,
)


logger_mosek = logging.getLogger("Mosek_logger")

dict_type_bounds = {
    mosek.boundkey.up: "up",
    mosek.boundkey.lo: "lo",
    mosek.boundkey.fx: "fx",
}


class ConstraintsFusion(CommonConstraints):
    """
    Class to handle the current constraint.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        model: mosek.fusion.Model = None,
        **kwargs,
    ):
        """
        Initialize the CurrentConstraint class.

        Parameters
        ----------
        task: mosek.Task
            The MOSEK task.
        index: int
            The index of the constraint.
        """
        super().__init__(
            indexes_matrices=indexes_matrices,
            indexes_variables=indexes_variables,
            **kwargs,
        )
        self.model = model

    def add_var(self, indice_i: int, indice_j: int, num_matrix: int, value: float):
        """
        Add a variable to the current constraint.

        Parameters
        ----------
        indice_i: int
            The index of the variable in the i-th position.
        indice_j: int
            The index of the variable in the j-th position.
        num_matrix: int
            The number of the matrix.
        value: float
            The value of the variable.
        name: str
            The name of the variable.
        """
        if value == 0:
            return

        if (indice_i, indice_j, num_matrix) in self.list_cstr[
            self.current_num_constraint
        ]["elements"].keys():
            if indice_i <= indice_j:
                self.list_cstr[self.current_num_constraint]["elements"][
                    (indice_j, indice_i, num_matrix)
                ] += value
            else:
                self.list_cstr[self.current_num_constraint]["elements"][
                    (indice_i, indice_j, num_matrix)
                ] += value

        else:
            if indice_i <= indice_j:  # DIAGONAL OR UPPER TRIANGLE
                self.list_cstr[self.current_num_constraint]["elements"][
                    (indice_j, indice_i, num_matrix)
                ] = value  # NO DIVISION BY 2 : FUSION API IS TAKING CARE OF DIAGONAL TERMS
            else:  # LOWER TRIANGLE
                self.list_cstr[self.current_num_constraint]["elements"][
                    (indice_i, indice_j, num_matrix)
                ] = value

    def add_to_task(self):
        """
        Add the constraint to the task.
        """
        logger_mosek.info(f"Adding {self.list_cstr} constraints to the task...")

        for ind_cstr in range(len(self.list_cstr)):
            # print("Adding constraint ", self.list_cstr[ind_cstr]["name"])

            res = sort_lists_by_first(
                self.list_cstr[ind_cstr]["num_matrix"],
                self.list_cstr[ind_cstr]["i"],
                self.list_cstr[ind_cstr]["j"],
                self.list_cstr[ind_cstr]["value"],
            )

            expression = Expr.constTerm(0.0)
            for num_matrix, i, j, value in zip(
                res[0],
                res[1],
                res[2],
                res[3],
            ):
                dim = self.indexes_matrices.get_shape_matrix(num_matrix)
                M = np.zeros((dim, dim))
                M[i, j] = value
                name = self.indexes_matrices.get_name_matrix(num_matrix)
                var = self.model.getVariable(name)

                expression = Expr.add(
                    expression,
                    Expr.sum(Expr.mulElm(Matrix.dense(M), var)),
                )
            if self.list_cstr[ind_cstr]["bound_type"] == mosek.boundkey.fx:
                lb = self.list_cstr[ind_cstr]["lb"]
                constraint = self.model.constraint(
                    self.list_cstr[ind_cstr]["name"],
                    expression,
                    Domain.equalsTo(lb),
                )
            elif self.list_cstr[ind_cstr]["bound_type"] == mosek.boundkey.up:
                ub = self.list_cstr[ind_cstr]["ub"]
                constraint = self.model.constraint(
                    self.list_cstr[ind_cstr]["name"],
                    expression,
                    Domain.lessThan(ub),
                )
            elif self.list_cstr[ind_cstr]["bound_type"] == mosek.boundkey.lo:
                lb = self.list_cstr[ind_cstr]["lb"]
                constraint = self.model.constraint(
                    self.list_cstr[ind_cstr]["name"],
                    expression,
                    Domain.greaterThan(lb),
                )

    def add_model(self, model: mosek.fusion.Model):
        """
        Add model to the objective.

        Parameters
        ----------
        model: mosek.fusion.Model
            The MOSEK model.
        """
        self.model = model
