#!/usr/bin/env python3

from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.parameters import IonicSpecies
from pyNN.morphology import NeuroMLMorphology, uniform
from pyNN.utility import get_simulator
from pyNN.utility.plotting import Figure, Panel


sim, options = get_simulator()

sim.setup(timestep=0.025)

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ArborTwoCellExample"
cell_class.ion_channels = {'na': sim.NaChannel, 'kdr': sim.KdrChannel, 'pas': sim.PassiveLeak}

soma = Segment(proximal=P(x=-3, y=0, z=0, diameter=6),
               distal=P(x=3, y=0, z=0, diameter=6),
               name="soma", id=0)

morphology = NeuroMLMorphology(NMLMorphology(segments=(soma,)))

cell_type = cell_class(
    morphology=morphology,
    ionic_species={
        "na": IonicSpecies("na", reversal_potential=50.0),
        "k": IonicSpecies("k", reversal_potential=-77.0)
    },
    cm=1.0,
    Ra=35.4,
    na={"conductance_density": uniform('soma', 0.120)},
    kdr={"conductance_density": uniform('soma', 0.036)},
    pas={
        "conductance_density": uniform('soma', 0.0003),
        "e_rev": -54.3
    }
)

initial_values = {
    "v": [-60, -50],
    "na.m": [0.09364195, 0.25081208],
    "na.h": [0.41815053, 0.15344321],
    "kdr.n": [0.39626825, 0.55081431]
}
cells = sim.Population(2, cell_type, initial_values=initial_values)

stim = [
    sim.DCSource(start=10, stop=12, amplitude=0.01),
    sim.DCSource(start=8, stop=14, amplitude=0.003),
]
stim[0].inject_into(cells[0:1], location="soma")
stim[1].inject_into(cells[1:2], location="soma")

cells.record("v", locations={"soma": "soma"}, sampling_interval=0.1)
cells.record(['na.m', 'na.h', 'kdr.n'], locations={'soma': 'soma'}, sampling_interval=0.1)
cells.record("spikes")

sim.run(30)

data = cells.get_data().segments[0]
#vm = data.filter(name="soma.v")
spikes = data.spiketrains

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes.multiplexed[1]:
        print("{:3.3f}".format(t))
else:
    print("no spikes")

print("Plotting results ...")

#original_data = np.loadtxt("arbor_two_cells_recipe.original_data")
#plt.plot(original_data[:, 0], original_data[:, 1], "b-", lw=3)
# plt.plot(vm.times, vm[:, 0], 'g-')
# plt.plot(vm.times, vm[:, 1], 'r-')
# plt.xlabel("t (ms)")
# plt.ylabel("v (mV)")
# plt.savefig(f"{options.simulator}_two_cells_PyNN.png")

#breakpoint()

Figure(
        Panel(data.filter(name="soma.v")[0],
              ylabel="Membrane potential, soma (mV)",
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
        title="Two single-compartment HH neurons",
        annotations=f"Simulated with {options.simulator.upper()}"
    ).save(f"{options.simulator}_two_cells_PyNN.png")