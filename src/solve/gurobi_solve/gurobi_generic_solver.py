from typing import List
import torch
from networks import ReLUNN
import mosek
import yaml
import time
import os
import logging
import sys
import gurobipy as gp
from gurobipy import GRB

import sys
from functools import partial


from ..generic_solver import Solver
from tools import get_project_path
from .callback import SolutionCallback, CallbackData, mycallback
from .analyser import analyze_model


logger_gurobi = logging.getLogger("Gurobi_logger")


class GurobiSolver(Solver):
    """
    A solver that uses MOSEK to solve the optimization problem.
    """

    def __init__(
        self,
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS

        self.env = gp.Env(empty=True)
        self.env.start()
        print("self name : ", self.name)
        self.m = gp.Model(self.name, env=self.env)

        self.K = self.network.K
        self.n = self.network.n
        self.W = self.network.W
        self.b = self.network.b

        self.constant = 0

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        params = Solver.parse_yaml(yaml_file)
        return cls(**params, **kwargs)

    def run_optimization(self, verbose: bool = False):
        self.compute_bounds(self.bounds_method)

        self.initiate_solver()
        self.add_variables()
        self.add_objective()
        self.add_constraints()
        self.print_solver_info(verbose)
        callback_data = CallbackData(self.m.getVars())
        callback_func = partial(
            mycallback, cbdata=callback_data, logfile=f"gurobi_{self.name}.log"
        )
        self.m.optimize()  # self.callback

        results = self.get_results(verbose)
        logger_gurobi.info("Time taken to solve: %s seconds", self.time_solving)
        self.write_model()
        return results

    def retrieve_z(self):
        z_values = {}

        for layer in range(self.K + 1 if self.LAST_LAYER else self.K):
            for neuron in range(self.n[layer]):
                if (layer, neuron) in self.stable_inactives_neurons:
                    continue
                z_values[(layer, neuron)] = self.z[(layer, neuron)].X
        return z_values

    def retrieve_beta(self):
        beta_values = {}
        for class_label in self.ytargets:
            if class_label != self.ytrue:
                beta_values[class_label] = self.beta[class_label].X
        return beta_values

    def get_adversarial_attack(self):
        z_values = self.retrieve_z()
        Sol = []
        for j in range(self.n[0]):
            Sol.append(z_values[(0, j)])
        return Sol

    def get_results(self, verbose: bool = False):
        """
        Get the results of the optimization.
        """
        self.time_solving = self.m.runtime
        self.nb_nodes = self.m.NodeCount
        logger_gurobi.info("Number of nodes explored: %s", self.nb_nodes)
        logger_gurobi.info("Time taken to solve: %s seconds", self.time_solving)
        logger_gurobi.info("Number of variables: %s", self.m.NumVars)
        logger_gurobi.info("Number of constraints: %s", self.m.NumConstrs)
        logger_gurobi.info("Number of non-zero coefficients: %s", self.m.NumNZs)
        logger_gurobi.debug("Status: %s", self.m.Status)
        if self.m.Status == GRB.OPTIMAL:
            opt = self.m.ObjVal
            print("Optimal objective value: ", opt)
            print("Constant to add : ", self.constant)
            self.opt = opt + self.constant
            logger_gurobi.debug(
                f"Optimal objective value (with added constant) for model {self.name}: %s",
                self.opt,
            )
            print("Optimal objective value (with added constant): ", self.opt)
            z_values = self.retrieve_z()
            logger_gurobi.debug(f"z values: {z_values}")
            if self.BETAS:
                beta_values = self.retrieve_beta()
                logger_gurobi.debug(f"beta values: {beta_values} ")
            else:
                beta_values = None
        elif self.m.Status == GRB.INFEASIBLE:
            logger_gurobi.error("Model is infeasible")
            # analyze_model(self.m)
        else:
            opt = self.m.ObjVal
            print("UNKNOWN STATUS : Optimal objective value: ", opt)

    def write_model(self):
        """
        Write the model to a file.
        """
        if self.folder_name is not None:

            self.m.write(
                get_project_path(f"{self.folder_name}/{self.name}/{self.name}.lp")
            )
            self.m.printStats()
            logger_gurobi.info(
                f"Model written to {self.folder_name}/{self.name}/{self.name}.lp"
            )
            self.m.setParam(
                "LogFile",
                get_project_path(f"{self.folder_name}/{self.name}/{self.name}.log"),
            )

    def initiate_solver(self, **parameters):
        """
        Initialize the solver with the given parameters.
        """
        self.m.Params.NonConvex = 2
        for key, value in parameters.items():
            if key in ["DualReductions"]:
                self.m.setParam(key, value)
            else:
                self.m.setParam(key, value)
        self.m.setParam("Threads", 1)

    def print_solver_info(self, verbose: bool = False):
        print("printing solver info : verbose = ", verbose)
        if verbose:
            self.env.setParam("OutputFlag", 1)
            self.m.setParam("LogToConsole", 1)
        else:
            self.env.setParam("OutputFlag", 0)

        if self.folder_name is not None:
            self.m.setParam(
                "LogFile",
                get_project_path(f"{self.folder_name}/{self.name}/Gurobi_logger.log"),
            )
        else:
            self.m.setParam("LogFile", get_project_path("results/Gurobi_logger.log"))

    def solve(self, verbose: bool = False):
        """
        Solve the optimization problem using MOSEK.
        """
        if "Lan" in self.__class__.__name__:
            for ytarget in self.ytargets:
                self.ytarget = ytarget
                self.run_optimization(verbose)
        else:
            self.run_optimization(verbose)

    def get_optimal_value(self):
        """
        Get the optimal value of the objective function.
        """
        if self.m.Status == GRB.OPTIMAL:
            return self.opt
        else:
            raise ValueError("Model is not optimal or has not been solved yet.")

    def add_variables(self):
        raise NotImplementedError(
            "add_variables method not implemented in GurobiSolver"
        )

    def add_objective(self):
        raise NotImplementedError(
            "add_objective method not implemented in GurobiSolver"
        )

    def add_constraints(self):
        raise NotImplementedError(
            "add_constraints method not implemented in GurobiSolver"
        )
