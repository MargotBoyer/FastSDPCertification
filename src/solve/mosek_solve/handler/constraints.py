from ast import Set
from tools import exists_two_similar_pairs_in_three_lists, deduct_two_lists
import logging
import numpy as np
import mosek
from numba.typed import Dict
import numba
from tools import infinity, deduplicate_and_sum, get_project_path

from .indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from .indexes_variables import Indexes_Variables_for_Mosek_Solver
from .variables_call import VariablesCall
from .variable_elements import (
    ElementsinConstraintsObjectives,
    add_dict_linear_to_elements,
)
import matplotlib.pyplot as plt

logger_mosek = logging.getLogger("Mosek_logger")


class CommonConstraints(VariablesCall):
    """
    Common functions for handling constraints in MOSEK.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        """
        Initialize the CommonConstraints class.

        Parameters
        ----------
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver
            The indexes of the matrices.
        indexes_variables: Indexes_Variables_for_Mosek_Solver
            The indexes of the variables.
        """
        super().__init__(
            indexes_matrices=indexes_matrices,
            indexes_variables=indexes_variables,
            **kwargs,
        )

        self.current_num_constraint = -1
        self.list_cstr = []
        self.cstr_names = set()

    def add_constant(self, value: float):
        """
        Add a constant to the constraint.
        """
        self.list_cstr[self.current_num_constraint]["constant"] += value

    def add_var(self, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def check_current_constraint(self):
        """
        Check if the current constraint is valid.
        """
        if self.current_num_constraint == -1:
            raise ValueError("No current constraint. Please create a new one.")
        else:
            name = self.list_cstr[self.current_num_constraint]["name"]
            print("Validating current constraint name : ", name)
            if self.list_cstr[self.current_num_constraint]["bound_type"] is None:
                raise ValueError(
                    "No bound type for the current constraint. Please set a bound type."
                )
            if self.list_cstr[self.current_num_constraint]["lb"] is None:
                raise ValueError(
                    "No lower bound for the current constraint. Please set a lower bound."
                )
            if self.list_cstr[self.current_num_constraint]["ub"] is None:
                raise ValueError(
                    "No upper bound for the current constraint. Please set an upper bound."
                )
            if self.list_cstr[self.current_num_constraint]["num_matrix"].size == 0:

                raise ValueError(
                    "No variable num_matrix for the current constraint. Please set a variable."
                )
            if self.list_cstr[self.current_num_constraint]["i"].size == 0:
                raise ValueError(
                    "No variable i for the current constraint. Please set a variable."
                )
            if self.list_cstr[self.current_num_constraint]["j"].size == 0:
                raise ValueError(
                    "No variable j for the current constraint. Please set a variable."
                )
            # exists, num_matrix, i, j = exists_two_similar_pairs_in_three_lists(
            #     self.list_cstr[self.current_num_constraint]["num_matrix"],
            #     self.list_cstr[self.current_num_constraint]["i"],
            #     self.list_cstr[self.current_num_constraint]["j"],
            # )
            # if exists:
            #     raise ValueError(
            #         f"Two similar pairs in the current constraint {name}: with the matrix n°{num_matrix} and indexes i={i} and j={j} \n \n {self.list_cstr[self.current_num_constraint]}"
            #     )
            # diff = deduct_two_lists(
            #     self.list_cstr[self.current_num_constraint]["j"],
            #     self.list_cstr[self.current_num_constraint]["i"],
            # )
            # if any(el > 0 for el in diff):
            #     index = next(
            #         (ind for ind, el in enumerate(diff) if el > 0),
            #         None,
            #     )
            #     i = self.list_cstr[self.current_num_constraint]["i"][index]
            #     j = self.list_cstr[self.current_num_constraint]["j"][index]
            #     raise ValueError(
            #         f"Indexes i and j are not sorted in the current constraint {name} : i = {i} and j = {j} \n"
            #     )

            # logger_mosek.info(
            #     f"Current constraint {self.current_num_constraint} is valid."
            # )
            # if any(
            #     el == 0 for el in self.list_cstr[self.current_num_constraint]["value"]
            # ):
            #     raise ValueError(
            #         f"Zero value in the current constraint {name} : {self.list_cstr[self.current_num_constraint]}"
            #     )
            if not (
                (
                    self.list_cstr[self.current_num_constraint]["i"].size
                    == self.list_cstr[self.current_num_constraint]["j"].size
                )
                or (
                    self.list_cstr[self.current_num_constraint]["num_matrix"].size
                    == self.list_cstr[self.current_num_constraint]["value"].size
                )
                or (
                    self.list_cstr[self.current_num_constraint]["i"].size
                    == self.list_cstr[self.current_num_constraint]["value"].size
                )
            ):
                raise ValueError(
                    f"Size mismatch in the current constraint {name} : {self.list_cstr[self.current_num_constraint]}"
                )

    def print_current_constraint(self):
        """
        Print the current constraint.
        """
        if self.current_num_constraint == -1:
            raise ValueError("No current constraint. Please create a new one.")
        else:
            name = self.list_cstr[self.current_num_constraint]["name"]
            elements = self.list_cstr[self.current_num_constraint]["elements"]
            ub = self.list_cstr[self.current_num_constraint]["ub"]
            lb = self.list_cstr[self.current_num_constraint]["lb"]
            bound_type = self.list_cstr[self.current_num_constraint]["bound_type"]
            # print(
            #     f"Current constraint {name} : {elements};     bound : {bound_type}; ub : {ub}; lb : {lb}\n"
            # )

    def add_bound(self, bound_type: mosek.boundkey, bound: float):
        """
        Add a bound to the constraint.

        Parameters
        ----------
        bound_type: str
            The type of the bound (lower or upper).
        **kwargs: Dict
            The keyword arguments for the bound : lb, ub
        """
        self.list_cstr[self.current_num_constraint]["bound_type"] = bound_type
        if bound_type == mosek.boundkey.fx:
            self.list_cstr[self.current_num_constraint]["lb"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
            self.list_cstr[self.current_num_constraint]["ub"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
        elif bound_type == mosek.boundkey.up:
            self.list_cstr[self.current_num_constraint]["lb"] = -infinity
            self.list_cstr[self.current_num_constraint]["ub"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
        elif bound_type == mosek.boundkey.lo:
            self.list_cstr[self.current_num_constraint]["lb"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
            self.list_cstr[self.current_num_constraint]["ub"] = infinity
        self.formate_cstr()

    def formate_cstr(self):
        """
        Format the constraint to be added to the task : adds values of parameters for the same variables.
        """
        print("Formatting constraint", self.current_num_constraint)

        i, j, num_matrix, val = self.list_cstr[self.current_num_constraint][
            "elements"
        ].decode_key_vec()

        self.list_cstr[self.current_num_constraint]["i"] = i
        self.list_cstr[self.current_num_constraint]["j"] = j
        self.list_cstr[self.current_num_constraint]["num_matrix"] = num_matrix
        self.list_cstr[self.current_num_constraint]["value"] = val

        print(f"Formatted constraint {self.current_num_constraint} ")

        # i_ = np.array(self.list_cstr[self.current_num_constraint]["i"])
        # j_ = np.array(self.list_cstr[self.current_num_constraint]["j"])
        # num_matrix = np.array(self.list_cstr[self.current_num_constraint]["num_matrix"])
        # val = np.array(self.list_cstr[self.current_num_constraint]["value"])

        # i = np.where(i_ > j_, i_, j_)
        # j = np.where(i_ > j_, j_, i_)
        # keys = np.stack((i, j, num_matrix), axis=-1)
        # unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        # values = np.bincount(inverse, weights=val)
        # i, j, num_matrix = unique_keys.T

        # self.list_cstr[self.current_num_constraint]["i"] = i
        # self.list_cstr[self.current_num_constraint]["j"] = j
        # self.list_cstr[self.current_num_constraint]["num_matrix"] = num_matrix
        # self.list_cstr[self.current_num_constraint]["value"] = values
        # print(
        #     f"Formatted constraint {self.current_num_constraint} : {self.list_cstr[self.current_num_constraint]}"
        # )

    def __str__(self):
        """
        String representation of the Constraints class.
        """
        line = "Constraints : \n"
        for i in range(len(self.list_cstr)):
            line += f"Constraint {i} : {self.list_cstr[i]}\n \n"

        return line

    def new_constraint(self, name: str, label: str = "to_change"):
        """
        Create a new constraint.
        """
        if self.current_num_constraint != -1:
            self.check_current_constraint()

        assert label in ["to_change", "same_for_data"], (
            "Label must be either 'to_change' or 'same_for_data'. "
            f"Got {label} instead."
        )

        # logger_mosek.info("Creating new constraint")
        if name in self.cstr_names:
            logger_mosek.warning(
                f"Constraint {name} already exists. Skipping creation."
            )
            return True

        self.current_num_constraint += 1
        self.cstr_names.add(name)

        self.list_cstr.append(
            {
                "name": name,
                "elements": ElementsinConstraintsObjectives(
                    self.indexes_variables.max_index,
                ),
                "constant": 0.0,
                "lb": None,
                "ub": None,
                "bound_type": None,
                "dual_value": None,
                "label": "to_change",
            }
        )
        print(f"Creating new constraint {self.current_num_constraint} : {name}")
        return False

    def first_term_equal_zero(self, num_matrices):
        """
        Set the first term of the constraint to zero.

        Parameters
        ----------
        num_matrices: int
            The number of matrices.
        """
        logger_mosek.info("Setting the first term of the matrix to zero")

        for num_matrix in range(num_matrices):
            name_matrix = self.indexes_matrices.get_name_matrix(num_matrix)
            if self.new_constraint(f"First term equal to zero of matrix {name_matrix}"):
                continue
            self.list_cstr[self.current_num_constraint]["elements"].add(
                i=0, j=0, num_matrix=num_matrix, value=1.0
            )
            self.add_bound(
                bound_type=mosek.boundkey.fx,
                bound=1.0,
            )

    def end_constraints(self):
        """
        End the constraints.
        """
        logger_mosek.info("Ending constraints")
        self.current_num_constraint += 1

    def get_histogram_of_coefficients(self):
        """
        Get a histogram of the coefficients of the constraints.
        """
        print("Getting histogram of coefficients...")
        histogram_coeff = {}
        min_coeff = infinity
        max_coeff = -infinity
        sum_coeff = 0.0

        histogram_bound = {}
        min_bound = infinity
        max_bound = -infinity
        sum_bound = infinity

        close_to_zero_total_coeff = infinity
        close_to_zero_total_bound = infinity

        comparaison_by_constraints = []

        for cstr in self.list_cstr:

            if cstr["bound_type"] is not None:
                if cstr["bound_type"] == mosek.boundkey.fx:
                    bound = cstr["lb"]
                elif cstr["bound_type"] == mosek.boundkey.up:
                    bound = cstr["ub"]
                elif cstr["bound_type"] == mosek.boundkey.lo:
                    bound = cstr["lb"]
                if bound < min_bound:
                    min_bound = bound
                if bound > max_bound:
                    max_bound = bound
                if abs(bound) < close_to_zero_total_bound and abs(bound) > 1e-25:
                    close_to_zero_total_bound = abs(bound)
                sum_bound += bound
                if bound in histogram_bound:
                    histogram_bound[bound] += 1
                else:
                    histogram_bound[bound] = 1

            greater_coeff = 0
            smaller_coef = infinity
            for value in cstr["value"]:
                if value < min_coeff:
                    min_coeff = value
                if value > max_coeff:
                    max_coeff = value
                if abs(value) < close_to_zero_total_coeff and abs(value) > 1e-25:
                    close_to_zero_total_coeff = abs(value)
                if abs(value) < smaller_coef and abs(value) > 1e-25:
                    smaller_coef = abs(value)
                if abs(value) > greater_coeff:
                    greater_coeff = abs(value)
                sum_coeff += value
                if value in histogram_coeff:
                    histogram_coeff[value] += 1
                else:
                    histogram_coeff[value] = 1

            comparaison_by_constraints.append(
                {
                    "greater_coeff": greater_coeff,
                    "smaller_coef": smaller_coef,
                    "bound": abs(bound) if cstr["bound_type"] is not None else None,
                }
            )

        return (
            histogram_coeff,
            min_coeff,
            max_coeff,
            sum_coeff
            / sum(len(self.list_cstr[i]["value"]) for i in range(len(self.list_cstr))),
            close_to_zero_total_coeff,
            histogram_bound,
            min_bound,
            max_bound,
            sum_bound / len(self.list_cstr),
            close_to_zero_total_bound,
            comparaison_by_constraints,
        )

    def reinitialize(self):
        """
        Reinitialize the constraints.
        """
        logger_mosek.info("Reinitializing constraints")

        self.list_ctsr = list(
            filter(lambda d: d["label"] == "same_for_data", self.list_cstr)
        )
        self.current_num_constraint = len(self.list_cstr) - 1
