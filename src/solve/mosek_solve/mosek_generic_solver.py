import pandas as pd
from typing import List
import torch
from networks import ReLUNN
import mosek
import yaml
import time
import os
import logging
import sys
import itertools
from pydantic import ValidationError
from typing import Dict

import matplotlib.pyplot as plt
import json


from tools import (
    get_project_path,
    create_folder,
    FullCertificationConfig,
    add_functions_to_class,
    add_row_from_dict
)
from ..generic_solver import Solver
from .handler.mosek_fusion import MosekFusionHandler
from .handler.mosek_classic.handler_classic import MosekClassicHandler
from .run_benchmark import create_all_cuts_to_test
from .get_variables import get_results, get_results_width_model


from tools import change_to_zero_negative_values


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    create_all_cuts_to_test, get_results, get_results_width_model
)
class MosekSolver(Solver):
    """
    A solver that uses MOSEK to solve the optimization problem.
    """

    def __init__(
        self,
        MATRIX_BY_LAYERS: bool = False,
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        BETAS_Z: bool = False,
        ZBAR: bool = False,
        use_fusion: bool = False,
        **kwargs,
    ):
        super().__init__(LAST_LAYER=LAST_LAYER, **kwargs)

        self.MATRIX_BY_LAYERS = MATRIX_BY_LAYERS
        self.keep_penultimate_actives = kwargs.get("keep_penultimate_actives", None)
        assert self.keep_penultimate_actives is not None
        print("use fusion in MosekGenericSolver: ", use_fusion)
        self.cuts = kwargs.get("cuts")
        self.all_combinations_cuts = kwargs.get("all_combinations_cuts", False)
        self.create_all_cuts_to_test()
        self.RLT_props = kwargs.get("RLT_props")

        self.use_fusion = use_fusion

        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR

        self.alpha_1 = kwargs.get("alpha_1")
        self.alpha_2 = kwargs.get("alpha_2")

        self.write_model = kwargs.get("write_model")

        logger_mosek.info(f"Model {self.__class__.__name__} initialized.")

        print("STUDY : init in Mosek Solver presque done")

        self.initiate_solver()
        self.only_width_model = False
        print("STUDY : init in Mosek Solver tout done")


    @staticmethod
    def parse_yaml_mosek(yaml_file):
        with open(yaml_file, "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            validated_config = FullCertificationConfig(**raw_config)
        except ValidationError as e:
            print(f"Erreur de validation du fichier YAML :\n{e}")
            raise

        return dict(
            cuts=validated_config.certification_problem.cuts,
            all_combinations_cuts=validated_config.certification_problem.all_combinations_cuts,
            RLT_props=validated_config.certification_problem.RLT_props,
        )

    def return_solutions(self):
        """
        Return the solutions of the optimization problem.
        """
        return self.handler.indexes_matrices.current_matrices_variables

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        params = Solver.parse_yaml(yaml_file)
        params_sdp = cls.parse_yaml_mosek(yaml_file)
        return cls(**params, **params_sdp, **kwargs)

    def initiate_solver(self):
        if self.use_fusion:
            self.handler = MosekFusionHandler(
                dataset=self.dataset,
                epsilon=self.epsilon,
                ytrue=self.ytrue,
                ytarget=self.ytarget,
                ytargets=self.ytargets,
                K=self.network.K,
                n=self.network.n,
                W=self.network.W,
                b=self.network.b,
                L=self.L,
                U=self.U,
                MATRIX_BY_LAYERS=self.MATRIX_BY_LAYERS,
                keep_penultimate_actives=self.keep_penultimate_actives,
                LAST_LAYER=self.LAST_LAYER,
                BETAS=self.BETAS,
                BETAS_Z=self.BETAS_Z,
                ZBAR=self.ZBAR,
                stable_inactives_neurons=self.stable_inactives_neurons,
                stable_actives_neurons=self.stable_actives_neurons,
                folder_name=self.folder_name,
                name=self.name,
            )
        else:
            self.handler = MosekClassicHandler(
                dataset=self.dataset,
                epsilon=self.epsilon,
                ytrue=self.ytrue,
                ytarget=self.ytarget,
                ytargets=self.ytargets,
                K=self.network.K,
                n=self.network.n,
                W=self.network.W,
                b=self.network.b,
                L=self.L,
                U=self.U,
                MATRIX_BY_LAYERS=self.MATRIX_BY_LAYERS,
                keep_penultimate_actives=self.keep_penultimate_actives,
                LAST_LAYER=self.LAST_LAYER,
                BETAS=self.BETAS,
                BETAS_Z=self.BETAS_Z,
                ZBAR=self.ZBAR,
                stable_inactives_neurons=self.stable_inactives_neurons,
                stable_actives_neurons=self.stable_actives_neurons,
                folder_name=self.folder_name,
                name=self.name,
            )

    def run_optimization(self, cuts: Dict, verbose: bool = False):
        try:
            self.handler.is_robust = False
            if verbose : 
                print("STUDY : RLT_prop in run_optimization: ", self.RLT_prop)
                print("STUDY : Beginnning of run_optimization with cuts: ", cuts)
            # self.handler.renew_solver()
            start_pretreatment_time = time.time()
            if verbose : 
                print("Initializing ENV...")
            self.handler.initiate_env(verbose)
            if verbose : 
                print("Intializing ENV : DONE.")
            self.handler.print_solver_info(verbose)
            if verbose :
                print("STUDY : Handler initialized.")
            self.add_objective()
            # print(
            #     "Objective indexes matrices: ",
            #     self.handler.Objective.list_indexes_matrixes,
            # )
            # print(
            #     "Objective indexes variables i: ",
            #     self.handler.Objective.list_indexes_variables_i,
            # )
            # print(
            #     "Objective indexes variables j: ",
            #     self.handler.Objective.list_indexes_variables_j,
            # )
            # print(
            #     "Objective indexes variables value: ",
            #     self.handler.Objective.list_values,
            # )
            if verbose : 
                print("STUDY ; Objective created.")
            self.handler.initialize_variables()
            self.handler.print_num_variables()
            if verbose : 
                print("STUDY : Variables initialized.")
                print("Adding constraints to the task...")
            time_1 = time.time()
            self.add_constraints(cuts)  # Constraints must be added after variables
            if verbose: 
                print("STUDY : Constraints added.")
            if self.only_width_model:
                print("STUDY : Only width model, getting results without optimization...")
                self.get_results(cuts, verbose)
                return False
            print("STUDY : not only width model, proceeding to optimization...")
            time_2 = time.time()

            # print(self.Constraints)
            
            self.handler.initialize_constraints()
            if verbose :
                print("STUDY : Constraints initialized.")
                print("STUDY: Number of constraints: ", len(self.handler.Constraints.list_cstr))
            # # STATISTICS ON PARAMETER VALUES
            # (
            #     histogram_coeff,
            #     min_coeff,
            #     max_coeff,
            #     mean_coeff,
            #     close_to_zero_total_coeff,
            #     histogram_bound,
            #     min_bound,
            #     max_bound,
            #     mean_bound,
            #     close_to_zero_total_bound,
            #     comparaison_by_constraints,
            # ) = self.handler.Constraints.get_histogram_of_coefficients()

            # actives_str = "use_actives" if self.use_active_neurons else "no_actives"
            # width = round((max_coeff - min_coeff) / 100, 5)
            # plt.bar(
            #     histogram_coeff.keys(),
            #     histogram_coeff.values(),
            #     width,
            #     color="g",
            # )
            # plt.savefig(
            #     get_project_path(
            #         f"{self.folder_name}/HISTOGRAM/{self.network_name}/{self.network_name}_{actives_str}_coefficients_histogram.png"
            #     )
            # )
            # tab = pd.DataFrame(comparaison_by_constraints)
            # tab.to_csv(
            #     get_project_path(
            #         f"{self.folder_name}/HISTOGRAM/{self.network_name}/{self.network_name}_{actives_str}_comparaison_by_constraints.csv"
            #     ),
            #     index=False,
            # )
            # json.dump(
            #     {
            #         "coefficients_histogram": {
            #             "min": min_coeff,
            #             "max": max_coeff,
            #             "mean": mean_coeff,
            #             "close_to_zero": close_to_zero_total_coeff,
            #         },
            #         "bounds_histogram": {
            #             "min": min_bound,
            #             "max": max_bound,
            #             "mean": mean_bound,
            #             "close_to_zero": close_to_zero_total_bound,
            #         },
            #     },
            #     open(
            #         get_project_path(
            #             f"{self.folder_name}/HISTOGRAM/{self.network_name}/{self.network_name}_{actives_str}_coefficients_sdp_stats.json"
            #         ),
            #         "w",
            #     ),
            # )



            self.handler.Objective.add_to_task()
            if verbose : 
                print("STUDY : Objective added to the task.")
            self.handler.Constraints.add_to_task()
            if verbose :
                print("STUDY : Constraints added to the task.")
            if self.write_model : 
                self.handler.write_model(
                    cuts,
                    RLT_prop=self.RLT_prop,
                    data_index=self.data_index,
                    ytarget=self.ytarget,
                )
            self.handler.define_objective_sense()
            if verbose :
                print("STUDY : Objective sense defined.")

            end_pretreatment_time = time.time()
            self.handler.time_pretreatment = (
                end_pretreatment_time - start_pretreatment_time
            )
            if verbose:
                print(
                    "STUDY : Pretreatment computing time: ",
                    self.handler.time_pretreatment,
                )
            start_time = time.time()
            self.handler.optimize()
            end_time = time.time()
            self.handler.time_solving = end_time - start_time
            # self.handler.write_model(cuts)
            logger_mosek.info(
                "Time taken to solve: %s seconds", self.handler.time_solving
            )
            # print("Tracker : ", self.handler.tracker.get_arrays())
            if verbose :
                print("CALLBACK : Getting results ...")
            time_results_start = time.time()

            results = self.get_results(cuts, verbose)
            time_results_end = time.time()
            if verbose :
                print(
                    "Time taken to get results: %s seconds",
                    time_results_end - time_results_start,
                )
                print("results obtained: ", results)
                print("x : ", self.x)
                print(
                    "self.use_inactive_neurons: ",
                    self.use_inactive_neurons,
                    "self.use_active_neurons: ",
                    self.use_active_neurons,
                )
                print(
                    "Inactive neurons : ",
                    self.stable_inactives_neurons,
                    "  Active neurons : ",
                    self.stable_actives_neurons,
                )
                print("is robust in run_optimization: ", self.handler.is_robust)
            
        except Exception as e:
            if verbose : 
                print("ERROR : An error occurred during optimization:", str(e))
            logger_mosek.error("An error occurred during optimization: %s", str(e))
            self.handler.is_robust = False
        finally:
            self.handler.cleanup_mosek()

    def solve(self, verbose: bool = False, only_bounds: bool = False):
        """
        Solve the optimization problem using MOSEK.
        """
        print("VERBOSE IN SOLVE : ", verbose)
        if self.is_trivially_solved or only_bounds:
            if verbose : 
                print("STUDY : Trivially solved problem, no need to run optimization.")
            self.get_results_trivially_solved()
            return True
        for cuts in self.cuts_to_test:
            if verbose :
                print("Testing cuts: ", cuts)

            if "Lan" in self.__class__.__name__:
                if verbose : 
                    print("CALLBACK ytargets : ", self.ytargets)
                for ytarget in self.ytargets:

                    for RLT_prop in self.RLT_props:

                        if verbose :
                            print(f"Testing RLT_prop for ytarget {ytarget} ! ", RLT_prop)
                        self.RLT_prop = RLT_prop
                        self.ytarget = ytarget
                        self.run_optimization(cuts, verbose)
                        if self.handler.is_robust:
                            if verbose :
                                print("Robust solution found for ytarget:", ytarget)
                            break
                        else:
                            print("No robust solution found for ytarget:", ytarget)
            else:
                for RLT_prop in self.RLT_props:
                    if verbose :
                        print(f"Testing RLT_prop ! ", RLT_prop)
                    self.RLT_prop = RLT_prop
                    self.run_optimization(cuts, verbose)
                    if self.handler.is_robust:
                        if verbose :
                            print("Robust solution found for RLT_prop:", RLT_prop)
                        break
                    else:
                        if verbose : 
                            print("No robust solution found for RLT_prop:", RLT_prop)

    def __str__(self):
        """
        String representation of the solver.
        """
        line = f"MosekSolver(K={self.network.K}, n={self.network.n} \n"
        line += f"  cuts={self.cuts} \n"
        line += f"  all_combinations_cuts={self.all_combinations_cuts} \n"
        line += self.handler.print_index_variables_matrices()
        line += "\n \n                   Weights : \n"
        for k in range(1, self.K + 1):
            line += f"  Layer {k} : \n"
            for j in range(self.n[k]):
                line += f"      Neuron {j} : \n"
                line += f"              W : {self.W[k-1][j]} \n"
                line += f"              b : {self.b[k-1][j]} \n\n"

        return line

    def add_constraints(self):
        """
        Add constraints to the task.
        """
        raise NotImplementedError(
            "The method add_constraints is not implemented in the base class."
        )

    def add_objective(self):
        """
        Add the objective function to the task.
        """
        raise NotImplementedError(
            "The method add_objective is not implemented in the base class."
        )
