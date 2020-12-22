###################################################
###     	Network definition		###
###################################################

from network_params import *
import scaling
from connectivity import FixedTotalNumberConnect
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np


class Network:

    def __init__(self, sim):
        return None

    def setup(self, sim):
        # Create matrix of synaptic weights
        self.w = create_weight_matrix()
        model = getattr(sim, 'IF_curr_exp')
        script_rng = NumpyRNG(seed=6508015, parallel_safe=parallel_safe)
        distr = RandomDistribution('normal', [V0_mean, V0_sd], rng=script_rng)

        # Create cortical populations
        self.pops = {}
        for layer in layers:
            self.pops[layer] = {}
            for pop in pops:
                self.pops[layer][pop] = sim.Population(int(N_full[layer][pop] *
                                                           N_scaling), model, cellparams=neuron_params)
                self.pops[layer][pop].initialize(v=distr)
                # Store whether population is inhibitory or excitatory
                self.pops[layer][pop].annotate(type=pop)
                this_pop = self.pops[layer][pop]

                # Spike recording
                if record_fraction:
                    num_spikes = int(round(this_pop.size * frac_record_spikes))
                else:
                    num_spikes = n_record
                this_pop[0:num_spikes].record('spikes')

                # Membrane potential recording
                if record_v:
                    if record_fraction:
                        num_v = int(round(this_pop.size * frac_record_v))

                    else:
                        num_v = n_record_v
                    this_pop[0:num_v].record('v')

        # Create thalamic population
        if thalamic_input:
            self.thalamic_population = sim.Population(
                thal_params['n_thal'],
                sim.SpikeSourcePoisson,
                {'rate': thal_params['rate'],
                 'start': thal_params['start'],
                 'duration': thal_params['duration']})

        # Compute DC input before scaling
        if input_type == 'DC':
            self.DC_amp = {}
            for target_layer in layers:
                self.DC_amp[target_layer] = {}
                for target_pop in pops:
                    self.DC_amp[target_layer][target_pop] = bg_rate * \
                        K_ext[target_layer][target_pop] * w_mean * neuron_params['tau_syn_E'] / \
                        1000.
        else:
            self.DC_amp = {'L23': {'E': 0., 'I': 0.},
                           'L4': {'E': 0., 'I': 0.},
                           'L5': {'E': 0., 'I': 0.},
                           'L6': {'E': 0., 'I': 0.}}

        # Scale and connect

        # In-degrees of the full-scale model
        K_full = scaling.get_indegrees()

        if K_scaling != 1:
            self.w, self.w_ext, self.K_ext, self.DC_amp = scaling.adjust_w_and_ext_to_K(
                K_full, K_scaling, self.w, self.DC_amp)
        else:
            self.w_ext = w_ext

        if sim.rank() == 0:
            print('w: %s' % self.w)

        for target_layer in layers:
            for target_pop in pops:
                target_index = structure[target_layer][target_pop]
                this_pop = self.pops[target_layer][target_pop]
                # External inputs
                if input_type == 'DC' or K_scaling != 1:
                    this_pop.set(i_offset=self.DC_amp[target_layer][target_pop])
                if input_type == 'poisson':
                    poisson_generator = sim.Population(this_pop.size,
                                                       sim.SpikeSourcePoisson, {
                                                           'rate': bg_rate * self.K_ext[target_layer][target_pop]})
                    conn = sim.OneToOneConnector()
                    syn = sim.StaticSynapse(weight=self.w_ext)
                    sim.Projection(poisson_generator, this_pop, conn,
                                   syn, receptor_type='excitatory')
                if thalamic_input:
                    # Thalamic inputs
                    if sim.rank() == 0:
                        print('creating thalamic connections to %s%s' % (target_layer, target_pop))
                    C_thal = thal_params['C'][target_layer][target_pop]
                    n_target = N_full[target_layer][target_pop]
                    K_thal = round(np.log(1 - C_thal) / np.log((n_target * thal_params['n_thal'] - 1.) /
                                                               (n_target * thal_params['n_thal']))) / n_target
                    FixedTotalNumberConnect(sim, self.thalamic_population,
                                            this_pop, K_thal, w_ext, w_rel * w_ext,
                                            d_mean['E'], d_sd['E'])
                # Recurrent inputs
                for source_layer in layers:
                    for source_pop in pops:
                        source_index = structure[source_layer][source_pop]
                        if sim.rank() == 0:
                            print('creating connections from %s%s to %s%s' %
                                  (source_layer, source_pop, target_layer, target_pop))
                        weight = self.w[target_index][source_index]
                        if source_pop == 'E' and source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                            w_sd = weight * w_rel_234
                        else:
                            w_sd = abs(weight * w_rel)
                        FixedTotalNumberConnect(sim, self.pops[source_layer][source_pop],
                                                self.pops[target_layer][target_pop],
                                                K_full[target_index][source_index] * K_scaling,
                                                weight, w_sd,
                                                d_mean[source_pop], d_sd[source_pop])


def create_weight_matrix():
    w = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
    for target_layer in layers:
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            for source_layer in layers:
                for source_pop in pops:
                    source_index = structure[source_layer][source_pop]
                    if source_pop == 'E':
                        if source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                            w[target_index][source_index] = w_234
                        else:
                            w[target_index][source_index] = w_mean
                    else:
                        w[target_index][source_index] = g * w_mean
    return w
