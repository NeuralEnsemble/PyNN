###################################################
###     	validation_microcircuit		###
###    a modification of the microcircuit.py    ###
###################################################

import network
import plotting
from neo.io import PyNNTextIO
import time
import pyNN
from network_params import *
import os
import sys
from sim_params import simulator_params, system_params
sys.path.append(system_params['backend_path'])
sys.path.append(system_params['pyNN_path'])
# import logging # TODO! Remove if it runs without this line


# prepare simulation
# logging.basicConfig() # TODO! Remove if it runs without this line
simulator = simulator_params.keys()[0]
exec('import pyNN.%s as sim' % simulator)
sim.setup(**simulator_params[simulator])

# Uncomment the two lines below when networkunit is installed
#import sciunit
#from networkunit.capabilities import ProducesSpikeTrains

# ===================SciUnit Interface=====================

# Uncomment the two lines below when networkunit is installed
# class Potjans2014Microcircuit( sciunit.Model,
#                               ProducesSpikeTrains ):
# Comment the line below when networkunit is installed


class Potjans2014Microcircuit():
    '''
    Use case:
    '''

    def __init__(self):
        # prepare simulation
        #exec('import pyNN.%s as sim' % simulator)
        sim.setup(**simulator_params[simulator])

    # The network takes a good amount of time so rather than make the user
    # wait to produce some capability it is better to create the network
    # when the capability is evoked
    def create_network(self):
        # create network
        start_netw = time.time()
        self.n = network.Network(sim)
        self.n.setup(sim)
        end_netw = time.time()
        if sim.rank() == 0:
            print('Creating the network took %g s' % (end_netw - start_netw,))

    def simulate(self):
        # simulate
        if sim.rank() == 0:
            print("Simulating...")
        start_sim = time.time()
        t = sim.run(simulator_params[simulator]['sim_duration'])
        end_sim = time.time()
        if sim.rank() == 0:
            print('Simulation took %g s' % (end_sim - start_sim,))

    def write_spiketrains(self):
        start_writing = time.time()
        for layer in self.n.pops:
            for pop in self.n.pops[layer]:
                io = PyNNTextIO(filename=system_params['output_path']
                                + "/spikes_" + layer
                                + '_' + pop
                                + '_' + str(sim.rank())
                                + ".txt")
                spikes = self.n.pops[layer][pop].get_data('spikes',
                                                          gather=False)
                for segment in spikes.segments:
                    io.write_segment(segment)
                    if record_v:
                        io = PyNNTextIO(filename=system_params['output_path']
                                        + "/vm_" + layer
                                        + '_' + pop
                                        + '_' + str(sim.rank())
                                        + ".txt")
                        vm = self.n.pops[layer][pop].get_data('v',
                                                              gather=False)
                        for segment in vm.segments:
                            try:
                                io.write_segment(segment)
                            except AssertionError:
                                pass
        end_writing = time.time()
        print("Writing data took %g s" % (end_writing - start_writing,))

    def plot_and_save(self):
        if create_raster_plot and sim.rank() == 0:
            # Numbers of neurons from which spikes were recorded
            n_rec = [[0] * n_pops_per_layer] * n_layers
            for layer, i in layers.items():
                for pop, j in pops.items():
                    if record_fraction:
                        n_rec[i][j] = round(N_full[layer][pop] * N_scaling * frac_record_spikes)
                    else:
                        n_rec[i][j] = n_record
            plotting.show_raster_bars(raster_t_min, raster_t_max, n_rec,
                                      frac_to_plot, system_params['output_path'] + '/')

    def create_results_directory(self):
        current_directory = os.getcwd()
        build_path = current_directory + os.sep + "results"
        if not os.path.exists("results"):
            os.makedirs(build_path)

    def produce_spiketrains(self):
        self.create_network()
        self.simulate()
        self.create_results_directory()
        self.write_spiketrains()
        sim.end()
        print("The model " + self.__class__.__name__ + " produces_spiketrains.")
