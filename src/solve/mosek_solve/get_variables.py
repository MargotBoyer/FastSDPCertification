import logging
import mosek
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List

from tools import get_project_path, create_folder, add_row_from_dict
from .run_benchmark import (
    compute_cuts_str,
    all_possible_cuts,
    print_solution_to_file_for_cb_solver,
    print_dual_variable_to_file_for_cb_solver,
)

logger_mosek = logging.getLogger("Mosek_logger")


def initialize_variables(self):
    """
    Add variables to the task.
    """
    logger_mosek.info("Initializing variables...")
    print("Initializing variables...")
    if self.BETAS_Z:
        logger_mosek.info("Model with betaz variables")
        if self.MATRIX_BY_LAYERS:
            logger_mosek.info("Model with matrices by layers")
            for k in range(self.K - 2):
                self.add_matrix_variable(
                    name=f"z_layers_{k}_{k+1}",
                    dim=1
                    + self.n[k]
                    + self.n[k + 1]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(layer=k)
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=k + 1
                    ),
                )
            if self.LAST_LAYER:
                logger_mosek.info("Model with last layer in solution matrices")
                self.add_matrix_variable(
                    name=f"z_layers_{self.K-2}_{self.K-1}",
                    dim=1
                    + self.n[self.K - 2]
                    + self.n[self.K - 1]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=self.K - 2
                    )
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=self.K - 1
                    ),
                )
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-1}_{self.K}_zbar_betas",
                        dim=1
                        + self.n[self.K - 1]
                        + self.n[self.K]
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-1}_{self.K}_betas",
                        dim=1
                        + self.n[self.K - 1]
                        + self.n[self.K]
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )

            else:
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-2}_{self.K-1}_zbar_betas",
                        dim=1
                        + self.n[self.K - 2]
                        + self.n[self.K - 1]
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 2
                        )
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name=f"z_layers_{self.K-2}_{self.K-1}_betas",
                        dim=1
                        + self.n[self.K - 2]
                        + self.n[self.K - 1]
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 2
                        )
                        - self.indexes_variables.get_number_pruned_neurons_on_layer(
                            layer=self.K - 1
                        ),
                    )
        else:
            if self.LAST_LAYER:
                logger_mosek.info("Model with last layer in solution matrices")
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_zbar_betas",
                        dim=1
                        + sum(self.n)
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_betas",
                        dim=1
                        + sum(self.n)
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1
                        ),
                    )
            else:
                if self.ZBAR:
                    logger_mosek.info("Model with zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_until_penultimate_zbar_betas",
                        dim=1
                        + sum(self.n[: self.K])
                        + 1
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1,
                        ),
                    )
                else:
                    logger_mosek.info("Model without zbar")
                    self.add_matrix_variable(
                        name="z_all_layers_until_penultimate_betas",
                        dim=1
                        + sum(self.n[: self.K])
                        + self.n[self.K]
                        - 1
                        - self.indexes_variables.get_number_pruned_neurons_before_layer(
                            layer=self.K - 1,
                        ),
                    )

    else:
        if self.MATRIX_BY_LAYERS:
            logger_mosek.info("Model with matrices by layers")
            for k in range(self.K - 1):
                self.add_matrix_variable(
                    name=f"z_layers_{k}_{k+1}",
                    dim=1
                    + self.n[k]
                    + self.n[k + 1]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(layer=k)
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=k
                        + 1  # Peut-être qu'il y a une erreur sur le layer choisi ici ou plus haut
                    ),
                )
            if self.LAST_LAYER:
                self.add_matrix_variable(
                    name=f"z_layers_{self.K-1}_{self.K}",
                    dim=1
                    + self.n[self.K - 1]
                    + self.n[self.K]
                    - self.indexes_variables.get_number_pruned_neurons_on_layer(
                        layer=self.K
                    ),
                )
        else:
            if self.LAST_LAYER:
                self.add_matrix_variable(
                    name="z_all_layers",
                    dim=1
                    + sum(self.n)
                    - self.indexes_variables.get_number_pruned_neurons_before_layer(
                        layer=self.K
                    ),
                )
            else:
                print("Adding z_all_layers until penultimate layer without betas")
                print("sum(self.n[: self.K]) : ", sum(self.n[: self.K]))
                print(
                    "self.indexes_variables.get_number_pruned_neurons_before_layer(self.K) : ",
                    self.indexes_variables.get_number_pruned_neurons_before_layer(
                        self.K
                    ),
                )
                self.add_matrix_variable(
                    name="z_all_layers",
                    dim=1
                    + sum(self.n[: self.K])
                    - self.indexes_variables.get_number_pruned_neurons_before_layer(
                        self.K
                    ),
                )
        if self.BETAS:
            self.add_vector_variable(name="betas", dim=self.n[self.K])


class Matrices_Solutions:
    def __init__(self):
        self._data = {}

    def add_value(self, cuts, value):
        """
        Ajoute une value à une configuration existante

        Args:
            cuts: Liste ou ensemble des coupes actives
            value: value à ajouter
        """
        key = frozenset(cuts)
        if key not in self._data:
            print(
                f"Ajout de la configuration {key} non presente avec la value de dim {value.shape}"
            )
            self._data[key] = value

        else:
            return ValueError(
                f"Configuration {key} déjà présente avec la value de dim {self._data[key].shape}"
            )

        return self

    def configurations_disponibles(self):
        """Retourne toutes les configurations de coupes enregistrées"""
        return [set(config) for config in self._data.keys()]

    def __getitem__(self, cuts):
        """Permet d'utiliser l'opérateur [] pour accéder aux configurations"""
        """
        Récupère les values pour une combinaison de coupes actives

        Args:
            cuts: Liste ou ensemble des coupes actives

        Returns:
            Liste de values associées ou liste vide si combinaison non trouvée
        """
        key = frozenset(cuts)
        if key not in self._data:
            print(f"Configuration {key} non trouvée")
            return ValueError(
                f"Configuration {key} non trouvée dans les configurations disponibles"
            )
        return self._data.get(key)

    def __contains__(self, cuts):
        """Permet d'utiliser l'opérateur 'in' pour vérifier si une configuration existe"""
        return frozenset(cuts) in self._data


def get_matrices_variables(self, cuts: List):
    """
    Get the matrices variables of the optimization problem.
    """
    if self.current_matrices_variables is None:
        raise ValueError(
            "No matrices variables found. Please initialize the variables first."
        )
    matrices = []
    for ind_solution in range(len(self.current_matrices_variables)):
        name_solution = self.current_matrices_variables[ind_solution]["name"]
        mat = self.current_matrices_variables[ind_solution]["value"][cuts]
        matrices.append(mat)
    return matrices


def compute_solutions(self, cuts: List, print_sol: bool = False):
    """
    Get the solutions and dual variables of the optimization problem.
    """
    cuts_str = compute_cuts_str(cuts)
    if print_sol:
        file_cb = open(
            get_project_path(f"{self.folder_name}/{self.name}/results_{cuts_str}.txt"),
            "w",
        )
        file_cb.write("Primal Solutions \n")
    for ind_solution in range(len(self.indexes_matrices.current_matrices_variables)):

        name_solution = self.indexes_matrices.current_matrices_variables[ind_solution][
            "name"
        ]
        dim = self.indexes_matrices.current_matrices_variables[ind_solution]["dim"]

        sol = self.get_solution(
            ind_solution=ind_solution, name_solution=name_solution, dim=dim
        )

        self.indexes_matrices.current_matrices_variables[ind_solution][
            "value"
        ].add_value(cuts, sol)

        # if verbose:
        #     logger_mosek.debug(
        #         f"Solution for {name_solution} of dimension {dim}: {mat}"
        #     )

        self.save_matrix_png(sol, name_solution=name_solution, cuts=cuts)
        self.save_matrix_csv(sol, name_solution=name_solution, cuts=cuts)

        if print_sol:
            print_solution_to_file_for_cb_solver(
                sol,
                index_matrix=ind_solution,
                dim=dim,
                file_cb=file_cb,
            )

    if print_sol:
        file_cb.write("Dual Solutions \n")
        self.get_dual_variables()
        print_dual_variable_to_file_for_cb_solver(
            list_cstr=self.Constraints.list_cstr, file_cb=file_cb
        )
        file_cb.close()


def get_results(self, cuts: List, verbose: bool = False):
    """
    Recuperation of optimization results
    """
    logger_mosek.info("Recuperation of optimization results...")
    logger_mosek.info("Verbose in get_results : %s", verbose)
    if self.only_width_model:
        print("STUDY : Only width model, getting width model results...")
        self.get_results_width_model(cuts, verbose)
        return
    status = self.handler.get_solution_status()

    print("Status of the solution: ", status)
    num_iterations = self.handler.get_num_iterations()
    logger_mosek.info("Number of iterations: %s", num_iterations)

    dic_benchmark = {
        "network": self.network_name,
        "model": self.name,
        "dataset": self.dataset_name,
        "data_index": self.data_index,
        "label": self.ytrue,
        "label_predicted": self.network.label(self.x),
        "target": self.ytarget if "Lan" in self.__class__.__name__ else None,
        "epsilon": self.epsilon,
        "status": status,
        "iterations": num_iterations,
        "time": self.handler.time_solving,
        "pretreatment_time": self.handler.time_pretreatment,
        "bound_time": self.compute_bounds_time,
        "MATRIX_BY_LAYERS": self.MATRIX_BY_LAYERS,
        "LAST_LAYER": self.LAST_LAYER,
        "USE_STABLE_ACTIVES": self.use_active_neurons,
        "USE_STABLE_INACTIVES": self.use_inactive_neurons,
        "Nb_stable_inactives": len(self.stable_inactives_neurons),
        "Nb_stable_actives": len(self.stable_actives_neurons),
        "Nb_constraints" : len(self.handler.Constraints.list_cstr)
    }
    dic_benchmark.update({cut: True for cut in cuts})
    if "RLT" in cuts:
        dic_benchmark.update({"RLT_prop": self.RLT_prop})

    if self.handler.is_status_optimal():
        print("CALLBACK : optimal status")
        dic_info_optimal_values = self.handler.add_all_infos_optimal_values_to_dic(
            cuts
        )
        dic_benchmark.update(dic_info_optimal_values)
        print("CALLBACK : dic_info_optimal_values: ", dic_info_optimal_values)

    elif self.handler.is_status_infeasible():
        print ("CALLBACK : infeasible status")
        if verbose:
            logger_mosek.debug("Primal or dual infeasibility certificate found.\n")
        self.handler.get_dual_variables()
        if True:
            file_cb = open(
                get_project_path(f"{self.folder_name}/{self.name}/results.txt"),
                "w",
            )
            file_cb.write("Dual Solutions \n")
            print_dual_variable_to_file_for_cb_solver(
                list_cstr=self.handler.Constraints.list_cstr, file_cb=file_cb
            )
            file_cb.close()
    elif self.handler.is_status_unknown():
        print ("CALLBACK : unknown status")
        logger_mosek.debug("Unknown solution status")
        try:
            dic_info_optimal_values = self.handler.add_all_infos_optimal_values_to_dic(
                cuts,
            )
            print("CALLBACK : dic_info_optimal_values: ", dic_info_optimal_values)
            dic_benchmark.update(dic_info_optimal_values)
        except Exception as e:
            print("ERROR in get_results : ", e)
            logger_mosek.critical("ERROR IN GETTING SOLUTIONS: %s", e)
            pass
    else:
        print ("CALLBACK : other status: ")
        if verbose:
            logger_mosek.debug("Other solution status")

    print("dic benchmark keys : ", dic_benchmark)
    if self.benchmark_dataframe is None:
        print("STUDY : self.benchmark is None ")
        self.benchmark_dataframe = pd.DataFrame(dic_benchmark, index=[0])
    else:
        print("STUDY : self.benchmark is not None ", self.benchmark_dataframe)
        self.benchmark_dataframe = add_row_from_dict(
            self.benchmark_dataframe, dic_benchmark
        )
    print("benchmark_dataframe   : ", self.benchmark_dataframe)


def get_results_width_model(self, cuts: List, verbose: bool = False):  
    print("STUDY : Recuperation of optimization results for width model...") 
    nb_constraints = len(self.handler.Constraints.list_cstr)
    nb_variables = self.handler.print_num_variables()
    dic_benchmark = {
        "network": self.network_name,
        "model": self.name,
        "dataset": self.dataset_name,
        "data_index": self.data_index,
        "label": self.ytrue,
        "label_predicted": self.network.label(self.x),
        "target": self.ytarget if "Lan" in self.__class__.__name__ else None,
        "epsilon": self.epsilon,
        "MATRIX_BY_LAYERS": self.MATRIX_BY_LAYERS,
        "LAST_LAYER": self.LAST_LAYER,
        "USE_STABLE_ACTIVES": self.use_active_neurons,
        "USE_STABLE_INACTIVES": self.use_inactive_neurons,
        "Nb_stable_inactives": len(self.stable_inactives_neurons),
        "Nb_stable_actives": len(self.stable_actives_neurons),
        "Nb_constraints": nb_constraints,
        "Nb_variables": nb_variables,
    }
    dic_benchmark.update({cut: True for cut in cuts})
    if "RLT" in cuts:
        dic_benchmark.update({"RLT_prop": self.RLT_prop})
    if self.benchmark_dataframe is None:
        self.benchmark_dataframe = pd.DataFrame(dic_benchmark, index=[0])
    else:
        self.benchmark_dataframe = add_row_from_dict(
            self.benchmark_dataframe, dic_benchmark
        )
    print("STUDY at the end of get_results_width_model: benchmark_dataframe   : ", self.benchmark_dataframe)


def save_matrix_png(self, mat, name_solution, cuts: List):
    """
    Save a 2D matrix as an image with values rounded to two decimal places.
    """
    cuts_str = compute_cuts_str(cuts)
    max_width = 200

    mat = np.array(mat)
    n_rows, n_cols = mat.shape
    model_dir = get_project_path(f"{self.folder_name}/{self.name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if n_rows < max_width:

        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(self.name, fontsize=14, pad=20)

        table_data = [
            [f"{mat[i, j]:.2f}" for j in range(n_cols)] for i in range(n_rows)
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(n_cols)))

        plt.savefig(
            os.path.join(
                model_dir,
                f"{name_solution}_{cuts_str}_solution.png",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def save_matrix_csv(self, mat, name_solution, cuts: List):
    """
    Save a the solution matrix as a CSV file with values rounded to two decimal places.
    """

    cuts_str = compute_cuts_str(cuts)

    model_dir = get_project_path(f"{self.folder_name}/{self.name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    mat = pd.DataFrame(np.round(mat, decimals=2))

    mat.to_csv(
        os.path.join(model_dir, f"{name_solution}_{cuts_str}.csv"),
        index=False,
    )
    plt.close()
