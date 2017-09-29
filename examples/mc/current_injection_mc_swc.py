"""
Injecting time-varying current into multi-compartment cells.


"""

import matplotlib
matplotlib.use("Agg")
from pyNN.morphology import load_morphology, uniform, random_section, apical_dendrites
from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel


# === Configure the simulator ================================================

sim, options = get_simulator()

sim.setup(timestep=0.025) #*ms)


# === Create neuron model template ===========================================

#morph = load_morphology("http://neuromorpho.org/dableFiles/kisvarday/CNG%20version/oi15rpy4-1.CNG.swc")  # todo: smart inference of morphology file format
morph = load_morphology("oi15rpy4-1.CNG.swc")

# need to specify nseg for dendrite

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ExampleMultiCompartmentNeuron"
cell_class.ion_channels = {'pas': sim.PassiveLeak, 'na': sim.NaChannel, 'kdr': sim.KdrChannel}

cell_type = cell_class(morphology=morph,
                       cm=1.0,    # allow to set per segment
                       Ra=500.0,  # allow to set per segment?
                       pas={"conductance_density": uniform('all', 0.0003),
                            "e_rev":-54.3},
                       na={"conductance_density": uniform('soma', 0.120),
                           "e_rev": 50.0},
                       kdr={"conductance_density": uniform('soma', 0.036),
                            "e_rev": -77.0}
                       )

#import pdb; pdb.set_trace()

# === Create a population with two cells ====================================

cells = sim.Population(2, cell_type, initial_values={'v': [-60.0, -70.0]})  #*mV})

# === Inject current into the soma of cell #0 and the dendrite of cell #1 ===

step_current = sim.DCSource(amplitude=5.0, start=50.0, stop=150.0)
step_current.inject_into(cells[0:1], location="soma")
#step_current.inject_into(cells[1:2], location=apical_dendrites(fraction_along=0.9))
#step_current.inject_into(cells[1:2], location=random(after_branch_point(3)(apical_dendrites))
step_current.inject_into(cells[1:2], location=random_section(apical_dendrites()))


# cells[0] --> ID - 1 cell
# cells[0:1] --> PopulationView containing 1 cell

# === Record from both compartments of both cells ===========================

cells.record('spikes')
cells.record(['na.m', 'na.h', 'kdr.n'], locations={'soma': 'soma'})
cells.record('v', locations={'soma': 'soma', 'dendrite': random_section(apical_dendrites())})

# === Run the simulation =====================================================

sim.run(200.0)


# === Plot recorded data =====================================================

data = cells.get_data().segments[0]  # this is a Neo Segment

#data = cells.soma.get_data().segments[0]  # this is a Neo Segment
#data = cells.get_data(sections=["soma"]).segments[0]  # this is a Neo Segment

# The segment contains one AnalogSignal per compartment and per recorded variable
# and one SpikeTrain per neuron

print("Spike times: {}".format(data.spiketrains))

Figure(
        Panel(data.filter(name='soma.v')[0],
              ylabel="Membrane potential, soma (mV)",
              yticks=True, ylim=(-80, 40)),
        Panel(data.filter(name='dendrite.v')[0],
              ylabel="Membrane potential, dendrite (mV)",
              yticks=True, ylim=(-80, 40)),
        Panel(data.filter(name='soma.na.m')[0],
              ylabel="m, soma",
              yticks=True, ylim=(0, 1)),
         Panel(data.filter(name='soma.na.h')[0],
               xticks=True, xlabel="Time (ms)",
               ylabel="h, soma",
               yticks=True, ylim=(0, 1)),
        Panel(data.filter(name='soma.kdr.n')[0],
              ylabel="n, soma",
              xticks=True, xlabel="Time (ms)",
              yticks=True, ylim=(0, 1)),
        title="Responses of two-compartment neurons to current injection",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save("current_injection_mc_swc.png")

sim.end()
