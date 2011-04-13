import sys
sys.path.append("/home/andrew/dev/nineml_all/trunk/lib9ml/python/examples/AL/")
sys.path.append("/home/andrew/dev/nineml_all/trunk/code_generation/nmodl/")
leaky_iaf = __import__("leaky_iaf")
coba_synapse = __import__("coba_synapse")
from pyNN.neuron.nineml import nineml_cell_type
from pyNN.utility import init_logging
import pyNN.neuron as sim

init_logging(None, debug=True)
sim.setup(timestep=0.1, min_delay=0.1)

celltype_cls = nineml_cell_type("if_cond_exp",
                                leaky_iaf.c1,
                                excitatory=coba_synapse.c1,
                                port_map=[('V', 'V'), ('Isyn', 'Isyn')])

parameters = {
    'C': 1.0,
    'gL': 50.0,
    't_ref': 5.0,
    'tau': 2.0,
    'theta': -50.0,
    'vL': -65.0,
    'V_reset': -65.0
}

#celltype = celltype_cls(parameters)
#cell = celltype.model(**celltype.parameters)

cells = sim.Population(1, celltype_cls, parameters)
cells.initialize('V', parameters['vL'])
cells.initialize('t_spike', -1e99) # neuron not refractory at start

input = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 100})

connector = sim.OneToOneConnector(weights=0.1, delays=0.5)
conn = sim.Projection(input, cells, connector, target='excitatory')

cells._record('V')
cells._record('g')
cells.record()

sim.run(100.0)

cells.recorders['V'].write("Results/nineml_neuron.V", filter=[cells[0]])
cells.recorders['g'].write("Results/nineml_neuron.g", filter=[cells[0]])

sim.end()
