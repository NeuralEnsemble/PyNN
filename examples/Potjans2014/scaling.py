#############################################################################
### Functions for computing and adjusting connection and input parameters ###
#############################################################################

import numpy as np
from network_params import *


def get_indegrees():
    '''Get in-degrees for each connection for the full-scale (1 mm^2) model'''
    K = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
    for target_layer in layers:
        for target_pop in pops:
            for source_layer in layers:
                for source_pop in pops:
                    target_index = structure[target_layer][target_pop]
                    source_index = structure[source_layer][source_pop]
                    n_target = N_full[target_layer][target_pop]
                    n_source = N_full[source_layer][source_pop]
                    K[target_index][source_index] = round(np.log(1. -
                                                                 conn_probs[target_index][source_index]) / np.log(
                        (n_target * n_source - 1.) / (n_target * n_source))) / n_target
    return K


def adjust_w_and_ext_to_K(K_full, K_scaling, w, DC):
    '''Adjust synaptic weights and external drive to the in-degrees
       to preserve mean and variance of inputs in the diffusion approximation'''
    K_ext_new = {}
    I_ext = {}
    for target_layer in layers:
        K_ext_new[target_layer] = {}
        I_ext[target_layer] = {}
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            x1 = 0
            for source_layer in layers:
                for source_pop in pops:
                    source_index = structure[source_layer][source_pop]
                    x1 += w[target_index][source_index] * K_full[target_index][source_index] * \
                        full_mean_rates[source_layer][source_pop]
            if input_type == 'poisson':
                x1 += w_ext*K_ext[target_layer][target_pop]*bg_rate
                K_ext_new[target_layer][target_pop] = K_ext[target_layer][target_pop]*K_scaling
            I_ext[target_layer][target_pop] = 0.001 * neuron_params['tau_syn_E'] * \
                (1. - np.sqrt(K_scaling)) * x1 + DC[target_layer][target_pop]
            w_new = w / np.sqrt(K_scaling)
            w_ext_new = w_ext / np.sqrt(K_scaling)
    return w_new, w_ext_new, K_ext_new, I_ext
