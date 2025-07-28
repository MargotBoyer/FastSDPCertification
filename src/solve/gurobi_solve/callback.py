import gurobipy as gp
from gurobipy import GRB
import logging
import time


logger_gurobi = logging.getLogger("Gurobi_logger")


class SolutionCallback:

    def __init__(self):
        self.time_threshold = 300
        self.start_time = time.time()
        self.print_time = time.time()
        self.target_gap_for_qp = 10
        self.target_gap_for_mip = 0.01
        self.previous_best_objective = None

    def __call__(self, model, where):
        raise NotImplementedError("This method should be implemented in subclasses.")


class NonConvexQuadraticProgramCallback(SolutionCallback):
    """
    Callback for non-convex quadratic programs.
    """

    def __init__(self, target_gap=0.01):
        super().__init__()

    def __call__(self, model, where):
        if where == GRB.Callback.MIP:

            # Récupérer le temps écoulé
            elapsed_time = time.time() - self.start_time
            print("Elapsed time: ", elapsed_time)

            try:
                # Pour les problèmes non convexes traités par branch-and-bound
                best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)

                # Vérifier si une solution a été trouvée
                try:
                    best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                    has_incumbent = True
                except:
                    best_obj = (
                        float("inf")
                        if model.ModelSense == GRB.MINIMIZE
                        else float("-inf")
                    )
                    has_incumbent = False

                # Calculer le gap si une solution existe
                if has_incumbent:
                    if abs(best_obj) < 1e-10:
                        rel_gap = abs(best_obj - best_bound)
                    else:
                        rel_gap = abs(best_obj - best_bound) / (1e-10 + abs(best_obj))
                else:
                    rel_gap = float("inf")

                # Récupérer le nombre de nœuds explorés
                node_count = model.cbGet(GRB.Callback.MIP_NODCNT)

                # Arrêter si temps suffisant ET gap acceptable sont atteints
                if (
                    elapsed_time > self.time_threshold
                    and has_incumbent
                    and rel_gap < self.target_gap_for_qp
                ):
                    print(
                        f"Arrêt anticipé - Nœuds: {node_count}, Temps: {elapsed_time:.2f}s"
                    )
                    if has_incumbent:
                        print(
                            f"Meilleure solution: {best_obj:.6f}, Meilleure borne: {best_bound:.6f}"
                        )
                        print(f"Gap relatif: {rel_gap:.6f}")
                    model.terminate()

                # Afficher périodiquement l'état (toutes les 30 secondes)
                if int(time.time() - self.print_time) >= 30:
                    status = "avec solution" if has_incumbent else "sans solution"
                    gap_str = f"Gap={rel_gap:.6f}" if has_incumbent else "Gap=N/A"
                    print(
                        f"Temps={elapsed_time:.2f}s, Nœuds={node_count}, {status}, {gap_str}"
                    )
                    self.print_time = time.time()

            except Exception as e:
                print("Exception dans le callback : ", e)
                # Si le callback échoue, on continue
                pass


#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

#   This example reads a model from a file, sets up a callback that
#   monitors optimization progress and implements a custom
#   termination strategy, and outputs progress information to the
#   screen and to a log file.
#
#   The termination strategy implemented in this callback stops the
#   optimization of a MIP model once at least one of the following two
#   conditions have been satisfied:
#     1) The optimality gap is less than 10%
#     2) At least 10000 nodes have been explored, and an integer feasible
#        solution has been found.
#   Note that termination is normally handled through Gurobi parameters
#   (MIPGap, NodeLimit, etc.).  You should only use a callback for
#   termination if the available parameters don't capture your desired
#   termination criterion.

import sys
from functools import partial


class CallbackData:
    def __init__(self, modelvars):
        self.modelvars = modelvars
        self.lastiter = -GRB.INFINITY
        self.lastnode = -GRB.INFINITY


def mycallback(model, where, *, cbdata, logfile):
    """
    Callback function. 'model' and 'where' arguments are passed by gurobipy
    when the callback is invoked. The other arguments must be provided via
    functools.partial:
      1) 'cbdata' is an instance of CallbackData, which holds the model
         variables and tracks state information across calls to the callback.
      2) 'logfile' is a writeable file handle.
    """

    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print(f"{cdels} columns and {rdels} rows are removed")
    elif where == GRB.Callback.SIMPLEX:
        # Simplex callback
        itcnt = model.cbGet(GRB.Callback.SPX_ITRCNT)
        if itcnt - cbdata.lastiter >= 100:
            cbdata.lastiter = itcnt
            obj = model.cbGet(GRB.Callback.SPX_OBJVAL)
            ispert = model.cbGet(GRB.Callback.SPX_ISPERT)
            pinf = model.cbGet(GRB.Callback.SPX_PRIMINF)
            dinf = model.cbGet(GRB.Callback.SPX_DUALINF)
            if ispert == 0:
                ch = " "
            elif ispert == 1:
                ch = "S"
            else:
                ch = "P"
            print(f"{int(itcnt)} {obj:g}{ch} {pinf:g} {dinf:g}")
    elif where == GRB.Callback.MIP:
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if nodecnt - cbdata.lastnode >= 100:
            cbdata.lastnode = nodecnt
            actnodes = model.cbGet(GRB.Callback.MIP_NODLFT)
            itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            print(
                f"{nodecnt:.0f} {actnodes:.0f} {itcnt:.0f} {objbst:g} "
                f"{objbnd:g} {solcnt} {cutcnt}"
            )
        if abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst)):
            print("Stop early - 10% gap achieved")
            model.terminate()
        if nodecnt >= 10000 and solcnt:
            print("Stop early - 10000 nodes explored")
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(cbdata.modelvars)
        print(
            f"**** New solution at node {nodecnt:.0f}, obj {obj:g}, "
            f"sol {solcnt:.0f}, x[0] = {x[0]:g} ****"
        )
    elif where == GRB.Callback.MIPNODE:
        # MIP node callback
        print("**** New node ****")
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(cbdata.modelvars)
            model.cbSetSolution(cbdata.modelvars, x)
    elif where == GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)
        print(f"{itcnt:.0f} {primobj:g} {dualobj:g} {priminf:g} {dualinf:g} {cmpl:g}")
    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        logfile.write(msg)
