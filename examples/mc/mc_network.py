# encoding: utf-8
"""
Some ideas for syntax for multicompartmental modelling in PyNN

Andrew Davison, May 2012
"""


import pyNN.neuron as sim
from pyNN.morphology import load_morphology, uniform, random_section, dendrites, apical_dendrites, by_distance
import neuroml.loaders ##
#from nineml.abstraction_layer.readers import XMLReader
from quantities import S, cm, um
from pyNN.space import Grid2D, RandomStructure, Sphere #, uniform, by_distance
from pyNN.parameters import IonicSpecies
from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel
#from neurom import longest_dendrite


# === Configure the simulator ================================================

sim, options = get_simulator()

sim.setup(timestep=0.025) #*ms)


pyr_morph = load_morphology("oi15rpy4-1.CNG.swc", replace_axon=None, use_library="morphio")

# # support ion channel models defined in NineML, LEMS, or from built-in library
# na_channel = XMLReader.read("na.xml")
# kdr_channel = load_lems("kd.xml")
# ka_channel = XMLReader.read("ka.xml")
# ampa = XMLReader.read("ampa.xml")
# gabaa = sim.ExpSynCond
#
# # first we define cell types (templates)

print("Building populations")

pyramidal_cell_class = sim.MultiCompartmentNeuron
pyramidal_cell_class.label = "PyramidalNeuron"
pyramidal_cell_class.ion_channels = {
      'pas': sim.PassiveLeak,
      'na': sim.NaChannel,
      'kdr': sim.KdrChannel
}
pyramidal_cell_class.post_synaptic_entities = {'AMPA': sim.CondExpPostSynapticResponse,
                                               'GABA_A': sim.CondExpPostSynapticResponse}

pyramidal_cell = pyramidal_cell_class(
    morphology=pyr_morph,
    pas={"conductance_density": uniform('all', 0.0003), "e_rev":-54.3},
    na={"conductance_density": uniform('soma', 0.120)},
    kdr={"conductance_density": uniform('soma', 0.036)},
    #kdr={"conductance_density": by_distance(apical_dendrites(), lambda d: 0.05*d/200.0)},
    ionic_species={
        "na": IonicSpecies("na", reversal_potential=50.0),
        "k": IonicSpecies("k", reversal_potential=-77.0)
    },
    cm=1.0,
    Ra=500.0,
    AMPA={
        "density": uniform('all', 0.05),  # number per µm
        "e_syn": 0.0,
        "tau_syn": 2.0
    },
    GABA_A={
        #"density": by_distance(dendrites(), lambda d: 0.05 * (d < 50.0)),  # number per µm
        "density": uniform('all', 0.05),
        "e_syn": -70.0,
        "tau_syn": 5.0
    }
)

# interneuron = ...
#
# now we actually create the cells in the simulator
pyramidal_cells = sim.Population(2, pyramidal_cell, initial_values={'v': -60.0}, structure=Grid2D())
# interneurons = sim.Population(1000, interneuron, structure=RandomStructure(boundary=Sphere(radius=300.0)))
inputs = sim.Population(1000, sim.SpikeSourcePoisson(rate=1000.0))

# inject current into soma of Pyramidal cells
##noise = sim.NoisyCurrentSource(mean=0.0, stdev=0.5, start=1.0, stop=9.0) # nA
##noise.inject_into(pyramidal_cells, location='soma')

# define which variables to record
#(pyramidal_cells + interneurons).record('spikes')    # record spikes from all cells
pyramidal_cells.record('spikes')
pyramidal_cells[:1].record('v', locations={"soma": "soma"})
pyramidal_cells[:1].record('v', locations={"dend": apical_dendrites()})
# interneurons.sample(20).record('v')                 # record soma.v from a sample of 20 granule cells
# interneurons[0:5].record('GABA_A.i', locations=['dendrite'])  # record the GABA_A synaptic current from the synapse
# pyramidal_cells[0].record('na.m', locations=longest_dendrite(pkj_morph)) # record the sodium channel m state variable along the length of one dendrite
#

print("Connecting populations")


# # connect populations
# weight_distr = sim.RandomDistribution('normal', (0.5, 0.1))
# depressing = sim.TsodysMarkramSynapse(U=500.0, weight=weight_distr, delay="0.2+d/100.0")
#
# p2g_connector = sim.DistanceDependentProbabilityConnector("0.1*exp(-d/100.0)",
#                                                           target="GABA_A")
#
# p2g = sim.Projection(pyramidal_cells, interneurons,
#                      connector=p2g_connector,
#                      synapse_type=depressing,
#                      source="soma.v")
#
# g2p = sim.Projection(interneurons, pyramidal_cells,
#                      sim.FromFileConnector("connections.h5"),
#                      sim.StaticSynapse(weight=weight_distr, delay=0.5),
#                      source="soma.v", target="AMPA")

i2p = sim.Projection(inputs, pyramidal_cells,
                     connector=sim.AllToAllConnector(location_selector=random_section(apical_dendrites())),
                     synapse_type=sim.StaticSynapse(weight=0.5, delay=0.5),
                     receptor_type="AMPA"
                     )


print("Running simulation")

sim.run(10)

# (pyramidal_cells + interneurons).write_data("output.h5")
data = pyramidal_cells.get_data().segments[0]

sim.end()


Figure(
        Panel(data.filter(name='soma.v')[0],
              ylabel="Membrane potential, soma (mV)",
              yticks=True, ylim=(-80, 40),
              xticks=True, xlabel="Time (ms)"),
        Panel(data.filter(name='dend.v')[0][:, 0::10],
              ylabel="Membrane potential, dendrites (mV)",
              yticks=True, ylim=(-80, 40),
              xticks=True, xlabel="Time (ms)"),
        title="Multi-compartment network",
        annotations="Simulated with NEURON"
    ).save(f"mc_network_{options.simulator}.png")

sim.end()