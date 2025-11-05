from typing import List
import numpy as np
import mosek
import os
import sys
import numba
from ..variable_elements import add_dict_linear_to_elements, add_dict_quad_to_elements


from ..indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from ..indexes_variables import Indexes_Variables_for_Mosek_Solver
from ..objective import Objective

import logging
import time


from mosek.fusion import Model, Expr, Matrix, ObjectiveSense
from tools import sort_lists_by_first


logger_mosek = logging.getLogger("Mosek_logger")


class ObjectiveFusion(Objective):
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
        Add a variable to the current objective.

        Parameters
        ----------
        indice_i: int
            The index of the variable in the i-th dimension.
        indice_j: int
            The index of the variable in the j-th dimension.
        num_matrix: int
            The index of the matrix.
        value: float
            The value of the variable.
        """
        self.list_indexes_matrixes.append(num_matrix)
        self.list_indexes_variables_i.append(indice_i)
        self.list_indexes_variables_j.append(indice_j)
        self.list_values.append(
            value
        )  # FUSION API IS TAKING CARE OF THE DIAGONAL ELEMENTS  

    def add_var(
        self, dict1: numba.typed.Dict, value: float, dict2: numba.typed.Dict = None
    ):
        """
        Add a variable to the current constraint.
        ----------
        """
        if dict2 is None:
            add_dict_linear_to_elements(
                elements=self.elements.elements,
                dict=dict1,
                value=value,
                nb_index=self.indexes_variables.max_index,
            )
        else:
            add_dict_quad_to_elements(
                elements=self.elements.elements,
                dict1=dict1,
                dict2=dict2,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_diag=True,
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

    def add_to_task(self):
        """
        Add objective to the task.
        """
        logging.info("Adding objective to the task")
        self.format_obj()


        expression = Expr.constTerm(self.constant)
        for num_matrix, i, j, value in zip(
            self.num_matrix,
            self.i,
            self.j,
            self.value,
        ):
            dim = self.indexes_matrices.get_shape_matrix(num_matrix)
            M = np.zeros((dim, dim))
            M[i, j] = value
            name = self.indexes_matrices.get_name_matrix(num_matrix)
            var = self.model.getVariable(name)

            # print(f"shape M : {M.numColumns}, {M.numRows}")
            M_matrix = Matrix.dense(M)
            expression = Expr.add(
                expression,
                Expr.sum(Expr.mulElm(M_matrix, var)),
            )
        self.model.objective(ObjectiveSense.Minimize, expression)
