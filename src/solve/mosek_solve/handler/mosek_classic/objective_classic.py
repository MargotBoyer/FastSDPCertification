from typing import List
import mosek
import os
import numpy as np
import sys
from ..indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from ..indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)
from ..variable_elements import (
    ElementsinConstraintsObjectives,
    add_dict_linear_to_elements,
    add_dict_quad_to_elements,
)
from ..objective import Objective
import logging
import time
import numba

from tools import divide_list_by


logger_mosek = logging.getLogger("Mosek_logger")


class ObjectiveClassic(Objective):
    """
    Class to handle the current constraint.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        task: mosek.Task = None,
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
        self.task = task

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

    def add_task(self, task: mosek.Task):
        """
        Add the task to the current constraint.

        Parameters
        ----------
        task: mosek.Task
            The MOSEK task.
        """
        self.task = task

    def add_to_task(self):
        """
        Add objective to the task.
        """
        logging.info("Adding Objective to the task")
        if self.verbose :
            print("Adding Objective to the task ...")
        self.format_obj()
        if self.verbose :
            print(
                f"Objective elements: {self.elements.elements}, num_matrix: {self.num_matrix}, i: {self.i}, j: {self.j}, value: {self.value}"
            )
            print("Objective size  : ", len(self.elements.elements))
            print(
                f"Objective i : {self.i.size}, j : {self.j.size}, num_matrix : {self.num_matrix.size}, value : {self.value.size}"
            )
        self.task.putbarcblocktriplet(
            self.num_matrix,
            self.i,
            self.j,
            self.value,
        )
        self.task.putobjsense(mosek.objsense.minimize)
