# coding: utf-8
"""
An implementation of benchmarks 1 and 2 from

    Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398

The IF network is based on the CUBA and COBA models of Vogels & Abbott
(J. Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).

The code is based on VAbenchmarks.py from PyNN/examples, and is modified
to test and plot the connection matrix of the network.

Usage: python connections.py <simulator> --plot-figure=name_figure

    <simulator> is either mock, neuron, nest, brian or pcsim

Joel Chavas, UNIC, CNRS
June 2014

"""

import os
import socket
from math import *

from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging, normalized_filename

from pyNN.random import NumpyRNG, RandomDistribution
from numpy import nan_to_num

   
def initialize():
    global sim
    global options
    global extra
    global rngseed
    global parallel_safe
    global rng
    global n_ext
    global n_exc
    global n_inh
    
    sim, options = get_simulator(
        ("--plot-figure", "Plot the connections to a file."))

    init_logging(None, debug=True)

    # === General parameters =================================================

    threads = 1
    rngseed = 98765
    parallel_safe = True
    rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)

    # === general network parameters (except connections) ====================

    n_ext = 20   # number of external stimuli
    n_exc = 60  # number of excitatory cells
    n_inh = 60  # number of inhibitory cells

    # === Options ============================================================

    extra = {'loglevel': 2, 'useSystemSim': True,
            'maxNeuronLoss': 0., 'maxSynapseLoss': 0.4,
            'hardwareNeuronSize': 8,
            'threads': threads,
            'filename': "connections.xml",
            'label': 'VA'}
    if sim.__name__ == "pyNN.hardware.brainscales":
        extra['hardware'] = sim.hardwareSetup['small']

    if options.simulator == "neuroml":
        extra["file"] = "connections.xml"
        
def build_connections(connector_type,connector_parameters):

    # === Setup ==============================================================

    node_id = sim.setup(**extra)
    np = sim.num_processes()

    host_name = socket.gethostname()
    print "Host #%d is on %s" % (node_id + 1, host_name)

    print "%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads'])

    # === Type references ====================================================
    celltype = sim.IF_cond_exp
    synapsetype = sim.StaticSynapse

    # === Definition of the types of neurons, synapses and connections =======
    progress_bar = ProgressBar(width=20)

    cell_stim = sim.SpikeSourceArray(spike_times=[0.0])
    cell_exc = celltype()
    cell_inh = celltype()
    syn_stim = synapsetype()
    syn_exc = synapsetype()
    syn_inh = synapsetype()
    conn_stim = connector_type(**connector_parameters)
    conn_exc = connector_type(**connector_parameters)
    conn_inh = connector_type(**connector_parameters)

    # === Populations ========================================================

    print "%s Creating cell populations..." % node_id
    pop_stim = sim.Population(n_ext, cell_stim, label="spikes")
    pop_exc = sim.Population(n_exc, cell_exc, label="Excitatory_Cells")
    pop_inh = sim.Population(n_inh, cell_inh, label="Inhibitory_Cells")

    print "%s Connecting populations..." % node_id

    connections = {}
    connections['stim2e'] = sim.Projection(
        pop_stim, pop_exc, connector=conn_stim, synapse_type=syn_stim, receptor_type='excitatory')
    connections['stim2i'] = sim.Projection(
        pop_stim, pop_inh, connector=conn_stim, synapse_type=syn_stim, receptor_type='excitatory')
    connections['e2e'] = sim.Projection(
        pop_exc, pop_exc, conn_exc, syn_exc, receptor_type='excitatory')
    connections['e2i'] = sim.Projection(
        pop_exc, pop_inh, conn_exc, syn_exc, receptor_type='excitatory')
    connections['i2e'] = sim.Projection(
        pop_inh, pop_exc, conn_inh, syn_inh, receptor_type='inhibitory')
    connections['i2i'] = sim.Projection(
        pop_inh, pop_inh, conn_inh, syn_inh, receptor_type='inhibitory')

    # === Output connection results ===========================================

    # for prj in connections.keys():
        # connections[prj].saveConnections('Results/VAconnections_%s_%s_np%d.conn'
        # % (prj, options.simulator, np))


    str_connections = "%d e->e  %d e->i  %d i->e  %d i->i" % (connections['e2e'].size(),
                                                            connections[
                                                                'e2i'].size(),
                                                            connections[
                                                                'i2e'].size(),
                                                            connections['i2i'].size())
    str_stim_connections = "%d stim->e  %d stim->i" % (
        connections['stim2e'].size(), connections['stim2i'].size())

    if node_id == 0:
        print "\n\n--- Test connections ---"
        print "Nodes                  : %d" % np
        print "Number of Neurons      : %d" % (n_exc+n_inh)
        print "Number of Synapses     : %s" % str_connections
        print "Number of inputs       : %s" % str_stim_connections
        print "\n"

    
    if options.plot_figure:
        filename = options.plot_figure + '_' + connector_type.__name__
        from pyNN.utility.plotting import Figure, Panel
        array_weights_exc = nan_to_num(
            connections['stim2e'].get('delay', format="array"))
        array_weights_inh = nan_to_num(
            connections['stim2i'].get('delay', format="array"))
        conn_weights_exc = nan_to_num(
            connections['e2e'].get('delay', format="array"))[0:20, :]
        conn_weights_inh = nan_to_num(
            connections['e2i'].get('delay', format="array"))[0:20, :]
        inh_weights_exc = nan_to_num(
            connections['i2e'].get('delay', format="array"))[0:20, :]
        inh_weights_inh = nan_to_num(
            connections['i2i'].get('delay', format="array"))[0:20, :]

        Figure(
            Panel(array_weights_exc, data_labels=["stim->exc"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
            Panel(array_weights_inh, data_labels=["stim->inh"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
            Panel(conn_weights_exc, data_labels=["exc->exc"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
            Panel(conn_weights_inh, data_labels=["exc->inh"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
            Panel(inh_weights_exc, data_labels=["inh->exc"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
            Panel(inh_weights_inh, data_labels=["inh->inh"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys'}]),
        ).save(filename)

    # === Finished with simulator ============================================

    sim.end()
    
if __name__ == "__main__":
    initialize()
    connector_type = [
        [ sim.FixedProbabilityConnector, {'p_connect':0.2, 'rng':rng} ],
        [ sim.AllToAllConnector, {'allow_self_connections': False} ]
        ]
    for conn in connector_type:
        build_connections(conn[0],conn[1])
    