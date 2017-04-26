"""
Injecting time-varying current into multi-compartment cells.


"""

import matplotlib
matplotlib.use("Agg")
from neuroml import Morphology, Segment, Point3DWithDiam as P
#from pyNN.units import uF_per_cm2, ohm_cm, S_per_cm2, mV, nA, ms
from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel


# === Configure the simulator ================================================

sim, options = get_simulator()

sim.setup(timestep=0.025) #*ms)


# === Create neuron model template ===========================================

soma = Segment(proximal=P(x=0, y=0, z=0, diameter=18.8),
               distal=P(x=18.8, y=0, z=0, diameter=18.8),
               name="soma")
dend = Segment(proximal=P(x=0, y=0, z=0, diameter=2),
               distal=P(x=-200, y=0, z=0, diameter=2),
               name="dendrite",
               parent=soma)

# need to specify nseg for dendrite

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ExampleMultiCompartmentNeuron"
cell_class.insert(pas=sim.PassiveLeak, sections=('soma', )) ###'dendrite'))  # or cell_class.whole_cell.pas = sim.PassiveLeak
cell_class.soma.insert(na=sim.NaChannel)  # or cell_class.soma.na = sim.NaChannel
cell_class.soma.insert(kdr=sim.KdrChannel) # or cell_class.soma.kdr = sim.KdrChannel

cell_type = cell_class(morphology=Morphology(segments=(soma, )), ###dend)),
                       cm=1.0, #*uF_per_cm2,   # allow to set per segment
                       Ra=123.0, #*ohm_cm)      # allow to set per segment?
                       #pas={"conductance_density": 0.0005, #*S_per_cm2)
                       #     "e_rev":-54.3},
                       na={"conductance_density": 0.120,
                           "e_rev": 50.0},
                       kdr={"conductance_density": 0.036,
                            "e_rev": -77.0}
                       )

#import pdb; pdb.set_trace()

# === Create a population with two cells ====================================

cells = sim.Population(2, cell_type, initial_values={'v': [-64.0, -65.0]})  #*mV})

# === Inject current into the soma of cell #0 and the dendrite of cell #1 ===

#step_current = sim.DCSource(amplitude=0.5*nA, start=50.0*ms, stop=400.0*ms)
step_current = sim.DCSource(amplitude=0.1, start=50.0, stop=150.0)
step_current.inject_into(cells[0:1], section="soma")
###step_current.inject_into(cells[1:2], section="dendrite")


# cells[0] --> ID - 1 cell
# cells[0:1] --> PopulationView containing 1 cell

# === Record from both compartments of both cells ===========================

#cells.record('spikes')
cells.record(['na.m', 'kdr.n'], sections=['soma'])
cells.record('v', sections=['soma']) ###, 'dendrite'])

# === Run the simulation =====================================================

sim.run(200.0)


# === Plot recorded data =====================================================

data = cells.get_data().segments[0]  # this is a Neo Segment

#data = cells.soma.get_data().segments[0]  # this is a Neo Segment
#data = cells.get_data(sections=["soma"]).segments[0]  # this is a Neo Segment

# The segment contains one AnalogSignal per compartment and per recorded variable
# and one SpikeTrain per neuron

Figure(
        Panel(data.filter(name='soma.v')[0],
              ylabel="Membrane potential, soma (mV)",
              #yticks=True, ylim=(-66*mV, -48*mV)),
              yticks=True), #, ylim=(-66, -48)),
        # Panel(data.filter(name='dendrite.v')[0],
        #       ylabel="Membrane potential, dendrite (mV)",
        #       #yticks=True, ylim=(-66*mV, -48*mV)),
        #       yticks=True), #, ylim=(-66, -48)),
        Panel(data.filter(name='soma.na.m')[0],
              ylabel="m, soma",
              yticks=True), #, ylim=(0, 1)),
        # Panel(data.filter(name='soma.kdr.n')[0],
        #       xticks=True, xlabel="Time (ms)",
        #       ylabel="n, soma",
        #       yticks=True), #, ylim=(0, 1)),
        Panel(data.filter(name='soma.kdr.n')[0],
              ylabel="n, soma",
              xticks=True, xlabel="Time (ms)",
              yticks=True),  # , ylim=(0, 1)),
        title="Responses of two-compartment neurons to current injection",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save("current_injection_mc.png")

#sim.end()
