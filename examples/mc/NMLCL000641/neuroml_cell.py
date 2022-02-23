"""


"""


import pyNN.neuron as sim

sim.setup()

cell_types = sim.neuroml.load_neuroml_cell_types("cADpyr229_L23_PC_c2e79db05a_0_0.cell.nml")
cell_type = cell_types[0]

population = sim.Population(1, cell_type())
