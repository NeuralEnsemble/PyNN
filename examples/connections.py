# coding: utf-8
"""
Examples of connections and plots of the connection matrices.

It loops over most connector types and plots the resulting connection matrices.

The network used is made of:
- a population of stimuli
- a population of inhibitory neurons
- a population of excitatory neurons

Usage: python connections.py <simulator> --plot-figure=name_figure

    <simulator> is either mock, neuron, nest, brian,...
    
It gives the results as png file, whose name is name_figure_ConnectorType, for each ConnectorType used.
The connection parameters used are defined in the __main__ loop at the bottom of the file

Joel Chavas, UNIC, CNRS
June 2014

"""

import os
import socket
from math import *

from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging, normalized_filename

from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.connectors import IndexBasedExpression
from numpy import nan_to_num, array, ones, savetxt

   
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

    n_ext = 60   # number of external stimuli
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
        
def build_connections(connector_type, connector_parameters):

    # === Setup ==============================================================

    node_id = sim.setup(**extra)
    np = sim.num_processes()

    host_name = socket.gethostname()
    print("Host #%d is on %s" % (node_id + 1, host_name))

    print("%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads']))

    # === Type references ====================================================
    celltype = sim.IF_cond_exp
    synapsetype = sim.StaticSynapse

    # === Definition of the types of neurons, synapses and connections =======
    progress_bar = ProgressBar(width=20)

    cell_stim = sim.SpikeSourceArray(spike_times=[1.0])
    cell_exc = celltype()
    cell_inh = celltype()
    syn_stim = synapsetype()
    syn_exc = synapsetype()
    syn_inh = synapsetype()
    conn_stim = connector_type(**connector_parameters)
    conn_exc = connector_type(**connector_parameters)
    conn_inh = connector_type(**connector_parameters)

    # === Populations ========================================================

    print("%s Creating cell populations..." % node_id)
    pop_stim = sim.Population(n_ext, cell_stim, label="spikes")
    pop_exc = sim.Population(n_exc, cell_exc, label="Excitatory_Cells")
    pop_inh = sim.Population(n_inh, cell_inh, label="Inhibitory_Cells")

    print("%s Connecting populations..." % node_id)

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
        print("\n\n--- Connector : %s ---"  % connector_type.__name__)
        print("Nodes                  : %d" % np)
        print("Number of Stims        : %d" % n_ext)
        print("Number of Exc Neurons  : %d" % n_exc)
        print("Number of Inh Neurons  : %d" % n_inh)
        print("Number of Synapses     : %s" % str_connections)
        print("Number of inputs       : %s" % str_stim_connections)
        print("\n")

    def normalize_array(arr):
        res = nan_to_num(arr)
        res = (res != 0)
        return res.astype(int)

    if options.plot_figure:
        filename = options.plot_figure + '_' + connector_type.__name__
        from pyNN.utility.plotting import Figure, Panel
        array_stim_exc = normalize_array(
            connections['stim2e'].get('delay', format="array")[0:20,:])
        array_stim_inh = normalize_array(
            connections['stim2i'].get('delay', format="array")[0:20,:])
        array_exc_exc = normalize_array(
            connections['e2e'].get('delay', format="array")[0:20,:])
        array_exc_inh = normalize_array(
            connections['e2i'].get('delay', format="array")[0:20,:])
        array_inh_exc = normalize_array(
            connections['i2e'].get('delay', format="array")[0:20,:])
        array_inh_inh = normalize_array(
            connections['i2i'].get('delay', format="array")[0:20,:])

        Figure(
            Panel(array_stim_exc, data_labels=["stim->exc"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys', 'vmin':0.}]),
            Panel(array_stim_inh, data_labels=["stim->inh"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys', 'vmin':0.}]),
            Panel(array_exc_exc, data_labels=["exc->exc"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys', 'vmin':0.}]),
            Panel(array_exc_inh, data_labels=["exc->inh"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys', 'vmin':0.}]),
            Panel(array_inh_exc, data_labels=["inh->exc"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys', 'vmin':0.}]),
            Panel(array_inh_inh, data_labels=["inh->inh"], line_properties=[
                {'xticks': True, 'yticks': True, 'cmap': 'Greys', 'vmin':0.}]),
        ).save(filename)

    # === Finished with simulator ============================================

    sim.end()

# ===========================================================================
# Utility functions
# ===========================================================================

def build_connection_parameters():
    global connection_list
    global path
    global array_connections
    
    connection_list = [
                (0, 0, 0.1, 0.1),
                (3, 0, 0.2, 0.11),
                (2, 3, 0.3, 0.12),
                (5, 1, 0.4, 0.13),
                (0, 1, 0.5, 0.14), 
                ]
    path = "test.connections"
    if os.path.exists(path):
        os.remove(path)
    savetxt(path, connection_list)

    array_connections = ones((60, 60), dtype=bool)
    array_connections[15, 15] = False
        
class IndexBasedProbability(IndexBasedExpression):
    def __call__(self, i, j):
        return array((i + j) % 3 == 0, dtype=float)        
    
def displacement_expression(d):
    return 0.5 * ((d[0] >= -1) * (d[0] <= 2)) + 0.25 * (d[1] >= 0) * (d[1] <= 1)


# ===========================================================================
# ===========================================================================
# MAIN PROGRAM
# ===========================================================================
# ===========================================================================

if __name__ == "__main__":
    
    # === Initializes =======================================================
    initialize()
    build_connection_parameters()
    
    # === Loop over connector types =========================================
    
    connector_type = [
        [ sim.FixedProbabilityConnector, {'p_connect':1.0, 'rng':rng} ],
        [ sim.AllToAllConnector, {'allow_self_connections': False} ],
        [ sim.DistanceDependentProbabilityConnector, {'d_expression':"exp(-abs(d))", 'rng':rng} ],
        [ sim.IndexBasedProbabilityConnector, {'index_expression': IndexBasedProbability(), 'rng':rng} ],
        [ sim.DisplacementDependentProbabilityConnector, {'disp_function': displacement_expression, 'rng':rng} ],
        [ sim.FromListConnector, {'conn_list': connection_list} ],
        [ sim.FromFileConnector, {'file': path, 'distributed': False} ],
        [ sim.FixedNumberPreConnector, {'n':3, 'rng': rng} ],
        [ sim.ArrayConnector, {'array': array_connections, 'safe': True} ]
        ]
    for conn in connector_type:
        build_connections(conn[0], conn[1])
    
