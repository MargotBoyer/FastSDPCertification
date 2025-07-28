from typing import List
import mosek
from ..indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from ..indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)

from ..constraints import CommonConstraints
from ..variable_elements import add_dict_linear_to_elements, add_dict_quad_to_elements
import os
import sys
import logging
import numpy as np
import numba


logger_mosek = logging.getLogger("Mosek_logger")

dict_type_bounds = {
    mosek.boundkey.up: "up",
    mosek.boundkey.lo: "lo",
    mosek.boundkey.fx: "fx",
}


class ConstraintsClassic(CommonConstraints):
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

        # A SUPPRIMER -- TEST
        # self.new_constraint(name="test")
        # self.add_linear_variable(
        #     var="z", layer=2, neuron=40, front_of_matrix=True, value=1
        # )
        # print(
        #     "Self stable actives neurons:",
        #     self.stable_actives_neurons + self.stable_inactives_neurons,
        # )
        # for neuron in range(self.n[self.K - 1]):
        #     if (self.K - 1, neuron) in self.stable_inactives_neurons or (
        #         self.K - 1,
        #         neuron,
        #     ) in self.stable_actives_neurons:
        #         print("Inactive  or active neuron:", neuron)
        #         continue

        #     self.add_quad_variable(
        #         var1="z",
        #         layer1=self.K - 1,
        #         neuron1=neuron,
        #         front_of_matrix1=False,
        #         var2="beta",
        #         class_label=2,
        #         value=1,
        #     )
        #     break
        # for neuron in range(self.n[self.K - 1]):
        #     if (self.K - 1, neuron) in self.stable_inactives_neurons or (
        #         self.K - 1,
        #         neuron,
        #     ) in self.stable_actives_neurons:
        #         print("Inactive or active  neuron:", neuron)
        #         continue

        #     self.add_quad_variable(
        #         var1="z",
        #         layer1=self.K - 1,
        #         neuron1=neuron,
        #         front_of_matrix1=False,
        #         var2="z",
        #         layer2=self.K - 1,
        #         neuron2=neuron,
        #         front_of_matrix2=False,
        #         value=1,
        #     )
        #     break

        # exit()

    def add_var(
        self, dict1: numba.typed.Dict, value: float, dict2: numba.typed.Dict = None
    ):
        """
        Add a variable to the current constraint.
        ----------
        """
        if dict2 is None:
            add_dict_linear_to_elements(
                elements=self.list_cstr[self.current_num_constraint][
                    "elements"
                ].elements,
                dict=dict1,
                value=value,
                nb_index=self.indexes_variables.max_index,
            )
        else:
            add_dict_quad_to_elements(
                elements=self.list_cstr[self.current_num_constraint][
                    "elements"
                ].elements,
                dict1=dict1,
                dict2=dict2,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_diag=True,
            )

    def add_task(self, task: mosek.Task):
        """
        Add the task to the constraint.

        Parameters
        ----------
        task: mosek.Task
            The MOSEK task.
        """
        self.task = task

    def add_to_task(self):
        """
        Add the constraint to the task.
        """
        logger_mosek.info(f"Adding {len(self.list_cstr)} constraints to the task...")
        for ind_cstr in range(len(self.list_cstr)):
            name = self.list_cstr[ind_cstr]["name"]
            i = self.list_cstr[ind_cstr]["i"]
            j = self.list_cstr[ind_cstr]["j"]
            num_matrix = self.list_cstr[ind_cstr]["num_matrix"]
            value = self.list_cstr[ind_cstr]["value"]
            print(
                f"Adding to task constraint {name} with num matrix: {num_matrix.size} , i= {i.size}, j = {j.size}, value = {value.size}"
            )
            assert (
                len(num_matrix) == len(i) == len(j) == len(value)
            ), "The length of num_matrix, i, j, and value must be the same."
            print(
                "Adding index :",
                [ind_cstr] * len(self.list_cstr[ind_cstr]["num_matrix"]),
            )
            print("type of num_matrix:", type(num_matrix))
            print("type of i:", type(i))
            print("type of j:", type(j))
            print("type of value:", type(value))

            self.task.putbarablocktriplet(
                ind_cstr * np.ones(len(self.list_cstr[ind_cstr]["num_matrix"]), dtype=np.int32),
                self.list_cstr[ind_cstr]["num_matrix"],
                self.list_cstr[ind_cstr]["i"],
                self.list_cstr[ind_cstr]["j"],
                self.list_cstr[ind_cstr]["value"],
            )
            self.task.putconbound(
                ind_cstr,
                self.list_cstr[ind_cstr]["bound_type"],
                self.list_cstr[ind_cstr]["lb"],
                self.list_cstr[ind_cstr]["ub"],
            )
