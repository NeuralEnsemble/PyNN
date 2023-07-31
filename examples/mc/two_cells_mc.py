"""
Two ball-and-stick cells with an excitatory synaptic connection
"""

import matplotlib
matplotlib.use("Agg")
from neuroml import Morphology, Segment, Point3DWithDiam as P
from pyNN.morphology import NeuroMLMorphology, uniform
from pyNN.parameters import IonicSpecies
#from pyNN.units import uF_per_cm2, ohm_cm, S_per_cm2, mV, nA, ms
from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel


# === Configure the simulator ================================================

sim, options = get_simulator()

sim.setup(timestep=0.025) #*ms)


# === Create neuron model template ===========================================

soma = Segment(proximal=P(x=18.8, y=0, z=0, diameter=18.8),
               distal=P(x=0, y=0, z=0, diameter=18.8),
               name="soma", id=0)
dend = Segment(proximal=P(x=0, y=0, z=0, diameter=2),
               distal=P(x=-500, y=0, z=0, diameter=2),
               name="dendrite",
               parent=soma, id=1)

# need to specify nseg for dendrite

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ExampleMultiCompartmentNeuron"
cell_class.ion_channels = {'pas': sim.PassiveLeak, 'na': sim.NaChannel, 'kdr': sim.KdrChannel}
cell_class.post_synaptic_entities = {'AMPA': sim.CondExpPostSynapticResponse}


cell_type = cell_class(
    morphology=NeuroMLMorphology(Morphology(segments=(soma, dend))),
    cm=1.0,    # mF / cm**2
    Ra=100.0,  # ohm.cm
    ionic_species={
            "na": IonicSpecies("na", reversal_potential=50.0),
            "k": IonicSpecies("k", reversal_potential=-77.0)
    },
    pas={"conductance_density": uniform('all', 0.0003), "e_rev":-54.3},
    na={"conductance_density": uniform('soma', 0.120)},
    kdr={"conductance_density": uniform('soma', 0.036)},
    AMPA={
        #"locations": random_placement(uniform('dendrite ', 0.05)),  # number per µm
        "locations": sim.morphology.at_distances("dendrite", [450]),
        "e_syn": 0.0,
        "tau_syn": 2.0
    },
)

# === Create a population with two cells ====================================

cells = sim.Population(2, cell_type, initial_values={'v': [-50.0, -65.0]})  #*mV})

# === Inject current into the soma of cell #0 ===

step_current = sim.DCSource(amplitude=0.2, start=30.0, stop=31.0)
step_current.inject_into(cells[0:1], location="soma")

# === Record from both compartments of both cells ===========================

cells.record('spikes')
cells.record(['na.m', 'na.h', 'kdr.n'], locations={'soma': 'soma'})
cells.record('v', locations={'soma': 'soma', 'dendrite': 'dendrite'})

# === Connect the two cells

print("Connecting neurons")
connections = sim.Projection(
    cells[0:1], cells[1:2],
    connector=sim.AllToAllConnector(),  #location_selector="dendrite"),
    synapse_type=sim.StaticSynapse(weight=0.2, delay=5.0),
    receptor_type="AMPA"
)

# === Run the simulation =====================================================

#breakpoint()
sim.run(50.0)


# === Plot recorded data =====================================================

data = cells.get_data().segments[0]  # this is a Neo Segment

#data = cells.soma.get_data().segments[0]  # this is a Neo Segment
#data = cells.get_data(sections=["soma"]).segments[0]  # this is a Neo Segment

# The segment contains one AnalogSignal per compartment and per recorded variable
# and one SpikeTrain per neuron

print("Spike times: {}".format(data.spiketrains))

xlim = (25, 50)
Figure(
        Panel(data.filter(name='soma.v')[0],
              ylabel="Membrane potential, soma (mV)",
              yticks=True, ylim=(-80, 40), xlim=xlim),
        Panel(data.filter(name='dendrite.v')[0],
              ylabel="Membrane potential, dendrite (mV)",
              yticks=True, ylim=(-80, -20), xlim=xlim),
        Panel(data.filter(name='soma.na.m')[0],
              ylabel="m, soma",
              yticks=True, ylim=(0, 1), xlim=xlim),
         Panel(data.filter(name='soma.na.h')[0],
               xticks=True, xlabel="Time (ms)",
               ylabel="h, soma",
               yticks=True, ylim=(0, 1), xlim=xlim),
        Panel(data.filter(name='soma.kdr.n')[0],
              ylabel="n, soma",
              xticks=True, xlabel="Time (ms)",
              yticks=True, ylim=(0, 1), xlim=xlim),
        title="Responses of two synaptically connected neurons to current injection",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(f"two_cells_mc_{options.simulator}.png")

sim.end()
