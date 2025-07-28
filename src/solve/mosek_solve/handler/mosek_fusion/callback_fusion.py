import sys
import numpy
from mosek.fusion import *
from mosek import callbackcode, iinfitem, dinfitem, liinfitem

def makeUserCallback(model, maxtime):
    def userCallback(caller,
                     douinf,
                     intinf,
                     lintinf):
        opttime = 0.0

        if caller == callbackcode.begin_intpnt:
            print("Callback : Starting interior-point optimizer")
        elif caller == callbackcode.intpnt:
            itrn = intinf[iinfitem.intpnt_iter]
            pobj = douinf[dinfitem.intpnt_primal_obj]
            dobj = douinf[dinfitem.intpnt_dual_obj]
            stime = douinf[dinfitem.intpnt_time]
            opttime = douinf[dinfitem.optimizer_time]

            print("Callback : Iterations: %-3d" % itrn)
            print("Callback :   Elapsed time: %6.2f(%.2f) " % (opttime, stime))
            print("Callback :   Primal obj.: %-18.6e  Dual obj.: %-18.6e" % (pobj, dobj))
        elif caller == callbackcode.end_intpnt:
            print("Callback : Interior-point optimizer finished.")
        elif caller == callbackcode.begin_primal_simplex:
            print("Callback : Primal simplex optimizer started.")
        elif caller == callbackcode.update_primal_simplex:
            itrn = intinf[iinfitem.sim_primal_iter]
            pobj = douinf[dinfitem.sim_obj]
            stime = douinf[dinfitem.sim_time]
            opttime = douinf[dinfitem.optimizer_time]

            print("Callback : Iterations: %-3d" % itrn)
            print("Callback :   Elapsed time: %6.2f(%.2f)" % (opttime, stime))
            print("Callback :   Obj.: %-18.6e" % pobj)
        elif caller == callbackcode.end_primal_simplex:
            print("Callback : Primal simplex optimizer finished.")
        elif caller == callbackcode.begin_dual_simplex:
            print("Callback : Dual simplex optimizer started.")
        elif caller == callbackcode.update_dual_simplex:
            itrn = intinf[iinfitem.sim_dual_iter]
            pobj = douinf[dinfitem.sim_obj]
            stime = douinf[dinfitem.sim_time]
            opttime = douinf[dinfitem.optimizer_time]
            print("Callback : Iterations: %-3d" % itrn)
            print("Callback :   Elapsed time: %6.2f(%.2f)" % (opttime, stime))
            print("Callback :   Obj.: %-18.6e" % pobj)
        elif caller == callbackcode.end_dual_simplex:
            print("Callback : Dual simplex optimizer finished.")
        elif caller == callbackcode.begin_bi:
            print("Callback : Basis identification started.")
        elif caller == callbackcode.end_bi:
            print("Callback : Basis identification finished.")
        else:
            pass

        if opttime >= maxtime:
            # mosek is spending too much time. Terminate it.
            print("Callback : Too much time, terminating.")
            return 1
        return 0

    return userCallback

def userProgresCallback(caller):
    # Handle the caller code here
    pass