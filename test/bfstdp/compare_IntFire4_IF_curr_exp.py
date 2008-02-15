
from pyNN.neuron import *

tau_m = 20.0
weight = 0.5
syn_delay = 0.5
tau_e = 5.0
cm = 3.18

setup(dt=0.0001, use_cvode=False, min_delay=syn_delay)

hoc_execute(['xopen("intfire4nc.hoc")'])

cell1 = create("IntFire4nc", {'taum': tau_m, 'taue': tau_e, 'taui1': 10, 'taui2': 15})
cell2 = create(IF_curr_exp, {'v_rest': 0, 'cm': cm, 'tau_m': tau_m,
                             'tau_refrac': 0.0, 'tau_syn_E': 5.0, 'tau_syn_I': 15.0,  
                             'i_offset': 0.0, 'v_reset': 0, 'v_thresh': 1.0,
                             'v_init': 0.0})

input = create(SpikeSourcePoisson, {'rate': 100.0, 'duration': 1000.0})

connect(input, cell1, weight=weight, synapse_type='syn', delay=syn_delay)
connect(input, cell2, weight=weight, synapse_type='excitatory', delay=syn_delay)

record_v(cell1, "compare_cell1.v")
record_v(cell2, "compare_cell2.v")
record(cell1, "compare_cell1.spikes")
record(cell2, "compare_cell2.spikes")

run(100.0)

end()

