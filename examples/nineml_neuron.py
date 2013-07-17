"""
Example of using a cell type defined in 9ML with pyNN.neuron
"""


import sys
import nineml.abstraction_layer as al
import pyNN.neuron as sim
from pyNN.neuron.nineml import nineml_celltype_from_model, CoBaSyn
from pyNN.utility import init_logging

from copy import deepcopy

init_logging(None, debug=True)
sim.setup(timestep=0.1, min_delay=0.1, max_delay=2.0)


iaf = al.ComponentClass(
    name="iaf",
    regimes=[
        al.Regime(
            name="subthresholdregime",
            time_derivatives=["dV/dt = ( gl*( vrest - V ) + ISyn)/(cm)"],
            transitions=al.On("V > vthresh",
                              do=["tspike = t",
                                  "V = vreset",
                                  al.OutputEvent('spikeoutput')],
                              to="refractoryregime"),
        ),

        al.Regime(
            name="refractoryregime",
            time_derivatives=["dV/dt = 0"],
            transitions=[al.On("t >= tspike + taurefrac",
                               to="subthresholdregime")],
        )
    ],
    state_variables=[
        al.StateVariable('V'),
        al.StateVariable('tspike'),
    ],
    analog_ports=[al.SendPort("V"),
                  al.ReducePort("ISyn", reduce_op="+"), ],

    event_ports=[al.SendEventPort('spikeoutput'), ],
    parameters=['cm', 'taurefrac', 'gl', 'vreset', 'vrest', 'vthresh']
)

coba = al.ComponentClass(
    name="CobaSyn",
    aliases=["I:=g*(vrev-V)", ],
    regimes=[
        al.Regime(
            name="cobadefaultregime",
            time_derivatives=["dg/dt = -g/tau", ],
            transitions=al.On('spikeinput', do=["g=g+q"]),
        )
    ],
    state_variables=[al.StateVariable('g')],
    analog_ports=[al.RecvPort("V"), al.SendPort("I"), ],
    parameters=['tau', 'q', 'vrev']
)

iaf_2coba = al.ComponentClass(
    name="iaf_2coba",
    subnodes={"iaf": iaf,
              "excitatory": coba,
              "inhibitory": deepcopy(coba)})
iaf_2coba.connect_ports("iaf.V", "excitatory.V")
iaf_2coba.connect_ports("iaf.V", "inhibitory.V")
iaf_2coba.connect_ports("excitatory.I", "iaf.ISyn")
iaf_2coba.connect_ports("inhibitory.I", "iaf.ISyn")

celltype_cls = nineml_celltype_from_model(
                    name="iaf_2coba",
                    nineml_model=iaf_2coba,
                    synapse_components = [
                        CoBaSyn(namespace='excitatory',  weight_connector='q'),
                        CoBaSyn(namespace='inhibitory',  weight_connector='q')]
                    )

parameters = {
    'iaf_cm': 1.0,
    'iaf_gl': 50.0,
    'iaf_taurefrac': 5.0,
    'iaf_vrest': -65.0,
    'iaf_vreset': -65.0,
    'iaf_vthresh': -50.0,
    'excitatory_tau': 2.0,
    'inhibitory_tau': 5.0,
    'excitatory_vrev': 0.0,
    'inhibitory_vrev': -70.0,
}

print celltype_cls.default_parameters

cells = sim.Population(1, celltype_cls, parameters)
cells.initialize(V=parameters['iaf_vrest'])
#cells.initialize(t_spike=-1e99) # neuron not refractory at start
#cells.initialize(regime=1002) # temporary hack

input = sim.Population(2, sim.SpikeSourcePoisson, {'rate': 100})

connector = sim.OneToOneConnector()
syn = sim.StaticSynapse(weight=1.0, delay=0.5)
conn = [sim.Projection(input[0:1], cells, connector, syn, receptor_type='excitatory'),
        sim.Projection(input[1:2], cells, connector, syn, receptor_type='inhibitory')]

cells.record(('spikes', 'iaf_V', 'excitatory_g', 'inhibitory_g'))

sim.run(100.0)

cells.write_data("Results/nineml_neuron.pkl")


sim.end()
