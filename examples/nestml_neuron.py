r"""
Example of using a cell type defined in NESTML
"""

import sys
from copy import deepcopy
from pyNN.utility import init_logging, get_simulator, normalized_filename
import pyNN
import pyNN.nest
import pyNN.nest.nestml



sim, options = get_simulator(("--plot-figure", "plot a figure with the given filename"))
init_logging(None, debug=True)
sim.setup(timestep=0.1, min_delay=0.1, max_delay=2.0)
celltype_cls = pyNN.nest.nestml.nestml_celltype_from_model(nestml_file_name="izhikevich_neuron.nestml")

parameters = {
    'a': .02,
    'b': .2,
    'c': -65.,
    'd': 8.
}

print(celltype_cls.default_parameters)

cells = sim.Population(1, celltype_cls, parameters)
cells.initialize(V_m=-70.)

input = sim.Population(2, sim.SpikeSourcePoisson, {'rate': 500})

connector = sim.OneToOneConnector()
syn = sim.StaticSynapse(weight=5.0, delay=0.5)
conn = [sim.Projection(input[0:1], cells, connector, syn)]

cells.record(('V_m', 'U_m'))

sim.run(100.0)

cells.write_data(
    normalized_filename("Results", "nestml_cell", "pkl",
                        options.simulator, sim.num_processes()),
    annotations={'script_name': __file__})

data = cells.get_data().segments[0]

sim.end()

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel

    Figure(
        Panel(data.filter(name='V_m')[0], ylabel="V_m", xlabel="Time (ms)", xticks=True),
        Panel(data.filter(name='U_m')[0], ylabel="U_m", xlabel="Time (ms)", xticks=True),
        title=__file__
    ).save(options.plot_figure)

    print(data.spiketrains)
