import numpy as np

from .certification_problem_constraints_bounds import McCormick_inter_layers
from tools import get_m_indexes_of_higher_values_in_list


# ********************************* RLT Lan Constraints *********************************
def add_RLT_constraints(self, p: float = 0.5):
    """
    Add the RLT constraints to the task.
    """
    print("Adding RLT constraint")
    for k in range(1, self.K + 1 if self.LAST_LAYER else self.K):
        nb_cstr = int(p * self.n[k - 1])
        print("RLT : number of neurones seleceted for layer", k, ":", nb_cstr)
        indexes_pruned = [
            j
            for j in range(self.n[k - 1])
            if (k - 1, j) in self.stable_inactives_neurons
            or (k - 1, j) in self.stable_actives_neurons
        ]
        print("Indexes pruned for layer", k, ":", indexes_pruned)
        for neuron_next in range(self.n[k]):
            if (k, neuron_next) in self.stable_inactives_neurons:
                print("RLT : neuron_next", neuron_next, "is stable, skipping")
                continue
            if (k, neuron_next) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                print("RLT : neuron_next", neuron_next, "is stable active, skipping")
                continue
            neurons_with_great_weights = get_m_indexes_of_higher_values_in_list(
                np.abs(self.W[k - 1][neuron_next]), nb_cstr, indexes_pruned
            )

            for neuron_prev in neurons_with_great_weights:
                # if (k - 1, neuron_prev) in self.stable_inactives_neurons or (
                #     k - 1,
                #     neuron_prev,
                # ) in self.stable_actives_neurons:
                #     continue
                assert (k - 1, neuron_prev) not in self.stable_inactives_neurons and (
                    k - 1,
                    neuron_prev,
                ) not in self.stable_actives_neurons
                print(
                    "Adding RLT constraint for neuron_prev:",
                    neuron_prev,
                    "neuron_next:",
                    neuron_next,
                )
                # assert self.U[k - 1][neuron_prev] > 0
                # assert self.U[k][neuron_next] > 0
                self.McCormick_inter_layers(k, neuron_prev, neuron_next)
