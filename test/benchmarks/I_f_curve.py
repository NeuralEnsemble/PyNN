"""
A single IF neuron with exponential, conductance-based synapses, fed by two
spike sources.

Run as:

$ python IF_cond_exp.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
May 2006

"""

from pyNN.utility import get_script_args, normalized_filename

simulator_name = get_script_args(1)[0]  
exec("import pyNN.%s as sim" % simulator_name)

sim.setup(timestep=0.1, min_delay=0.1, max_delay=4.0)

ifcell  = sim.IF_cond_exp(
    cm=0.2, i_offset=0.0, tau_refrac=3.0, v_thresh=-51.0, 
    tau_syn_E=5.0, tau_syn_I=5.0, v_reset=-70.0, e_rev_E=0., 
    e_rev_I=-100., v_rest=-50., tau_m=20.)

popcell = sim.Population(1,ifcell)

current_source = sim.DCSource(amplitude=2.0)
popcell.inject(current_source)

filename = normalized_filename("Results", "IF_cond_exp", "pkl", simulator_name)
sim.record(['v', 'gsyn_exc', 'gsyn_inh'], popcell, filename,
       annotations={'script_name': __file__})

sim.run(200.0)

sim.end()
