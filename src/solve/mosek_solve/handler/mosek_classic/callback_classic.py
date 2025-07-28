import mosek


callback_codes = [
    "begin_bi",
    "begin_conic",
    "begin_dual_bi",
    "begin_dual_sensitivity",
    "begin_dual_setup_bi",
    "begin_dual_simplex",
    "begin_dual_simplex_bi",
    "begin_folding",
    "begin_folding_bi",
    "begin_folding_bi_dual",
    "begin_folding_bi_initialize",
    "begin_folding_bi_optimizer",
    "begin_folding_bi_primal",
    "begin_infeas_ana",
    "begin_initialize_bi",
    "begin_intpnt",
    "begin_license_wait",
    "begin_mio",
    "begin_optimize_bi",
    "begin_optimizer",
    "begin_presolve",
    "begin_primal_bi",
    "begin_primal_repair",
    "begin_primal_sensitivity",
    "begin_primal_setup_bi",
    "begin_primal_simplex",
    "begin_primal_simplex_bi",
    "begin_qcqo_reformulate",
    "begin_read",
    "begin_root_cutgen",
    "begin_simplex",
    "begin_solve_root_relax",
    "begin_to_conic",
    "begin_write",
    "conic",
    "decomp_mio",
    "dual_simplex",
    "end_bi",
    "end_conic",
    "end_dual_bi",
    "end_dual_sensitivity",
    "end_dual_setup_bi",
    "end_dual_simplex",
    "end_dual_simplex_bi",
    "end_folding",
    "end_folding_bi",
    "end_folding_bi_dual",
    "end_folding_bi_initialize",
    "end_folding_bi_optimizer",
    "end_folding_bi_primal",
    "end_infeas_ana",
    "end_initialize_bi",
    "end_intpnt",
    "end_license_wait",
    "end_mio",
    "end_optimize_bi",
    "end_optimizer",
    "end_presolve",
    "end_primal_bi",
    "end_primal_repair",
    "end_primal_sensitivity",
    "end_primal_setup_bi",
    "end_primal_simplex",
    "end_primal_simplex_bi",
    "end_qcqo_reformulate",
    "end_read",
    "end_root_cutgen",
    "end_simplex",
    "end_simplex_bi",
    "end_solve_root_relax",
    "end_to_conic",
    "end_write",
    "folding_bi_dual",
    "folding_bi_optimizer",
    "folding_bi_primal",
    "heartbeat",
    "im_dual_sensivity",
    "im_dual_simplex",
    "im_license_wait",
    "im_lu",
    "im_mio",
    "im_mio_dual_simplex",
    "im_mio_intpnt",
    "im_mio_primal_simplex",
    "im_order",
    "im_primal_sensivity",
    "im_primal_simplex",
    "im_read",
    "im_root_cutgen",
    "im_simplex",
    "intpnt",
    "new_int_mio",
    "optimize_bi",
    "primal_simplex",
    "qo_reformulate",
    "read_opf",
    "read_opf_section",
    "restart_mio",
    "solving_remote",
    "update_dual_bi",
    "update_dual_simplex",
    "update_dual_simplex_bi",
    "update_presolve",
    "update_primal_bi",
    "update_primal_simplex",
    "update_primal_simplex_bi",
    "update_simplex",
    "write_opf",
]


def makeUserCallback(maxtime, task):
    pass

    def userCallback(caller, douinf, intinf, lintinf):
        opttime = 0.0

        code = callback_codes[caller]
        print("CODE ?  : ", code)
        if caller == mosek.callbackcode.begin_optimizer:
            print("CALLBACK : Starting optimizer")
        elif caller == mosek.callbackcode.end_optimizer:
            print("CALLBACK : Optimizer finished.")
        # INTERIOR-POINT
        elif caller == mosek.callbackcode.begin_intpnt:
            print("CALLBACK : Starting interior-point optimizer")
        elif caller == mosek.callbackcode.intpnt:
            itrn = intinf[mosek.iinfitem.intpnt_iter]
            pobj = douinf[mosek.dinfitem.intpnt_primal_obj]
            dobj = douinf[mosek.dinfitem.intpnt_dual_obj]
            stime = douinf[mosek.dinfitem.intpnt_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]

            print("CALLBACK  Iterations: %-3d" % itrn)
            print("CALLBACK   Elapsed time: %6.2f(%.2f) " % (opttime, stime))
            print("CALLBACK   Primal obj.: %-18.6e  Dual obj.: %-18.6e" % (pobj, dobj))
        elif caller == mosek.callbackcode.end_intpnt:
            print("CALLBACK : Interior-point optimizer finished.")

        # SIMPLEX
        elif caller == mosek.callbackcode.begin_simplex:
            print("CALLBACK  : Starting simplex optimizer")
        elif caller == mosek.callbackcode.begin_primal_simplex:
            print("CALLBACK  :Primal simplex optimizer started.")
        elif caller == mosek.callbackcode.update_primal_simplex:
            itrn = intinf[mosek.iinfitem.sim_primal_iter]
            pobj = douinf[mosek.dinfitem.sim_obj]
            stime = douinf[mosek.dinfitem.sim_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]

            print("CALLBACK  Iterations: %-3d" % itrn)
            print("CALLBACK   Elapsed time: %6.2f(%.2f)" % (opttime, stime))
            print("CALLBACK   Obj.: %-18.6e" % pobj)
        elif caller == mosek.callbackcode.end_primal_simplex:
            print("CALLBACK  Primal simplex optimizer finished.")
        elif caller == mosek.callbackcode.begin_dual_simplex:
            print("CALLBACK  Dual simplex optimizer started.")
        elif caller == mosek.callbackcode.update_dual_simplex:
            itrn = intinf[mosek.iinfitem.sim_dual_iter]
            pobj = douinf[mosek.dinfitem.sim_obj]
            stime = douinf[mosek.dinfitem.sim_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]
            print("CALLBACK  Iterations: %-3d" % itrn)
            print("CALLBACK   Elapsed time: %6.2f(%.2f)" % (opttime, stime))
            print("CALLBACK   Obj.: %-18.6e" % pobj)
        elif caller == mosek.callbackcode.end_dual_simplex:
            print("CALLBACK Dual simplex optimizer finished.")

        # CONIC
        elif caller == mosek.callbackcode.begin_conic:
            print("CALLBACK  CALLBACK : Starting conic optimizer")
        elif caller == mosek.callbackcode.conic:

            itrn = intinf[mosek.iinfitem.intpnt_iter]
            print("CALLBACK  Iterations: %-3d" % itrn)

            pobj = douinf[mosek.dinfitem.intpnt_primal_obj]
            dobj = douinf[mosek.dinfitem.intpnt_dual_obj]
            print("CALLBACK   Primal obj.: %-18.6e  Dual obj.: %-18.6e" % (pobj, dobj))

            # stime = douinf[mosek.dinfitem.conic_time]
            opttime = douinf[mosek.dinfitem.optimizer_time]

            print("CALLBACK   Elapsed time: %-18.6e" % opttime)
        elif caller == mosek.callbackcode.end_conic:
            print("CALLBACK : Conic optimizer finished.")

        # PRESOLVE
        elif caller == mosek.callbackcode.begin_presolve:
            print("CALLBACK : Starting presolve")
        elif caller == mosek.callbackcode.heartbeat:
            print("CALLBACK : Presolve heartbeat.")
        elif caller == mosek.callbackcode.end_presolve:
            print("CALLBACK : Presolve finished.")

        # MIO
        elif caller == mosek.callbackcode.new_int_mio:
            print("CALLBACK  New integer solution has been located.")
            xx = task.getxx(mosek.soltype.itg)
            print(xx)
            print("CALLBACK  Obj.: %f" % douinf[mosek.dinfitem.mio_obj_int])
        else:
            print("CALLBACK  Unknown callback event: %d" % caller)
            pass

        if opttime >= maxtime:
            # mosek is spending too much time. Terminate it.
            print("CALLBACK   Terminating.")
            return 1

        return 0

    return userCallback
