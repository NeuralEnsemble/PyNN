#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import numpy as np
import matplotlib.pyplot as plt
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.parameters import IonicSpecies
import pyNN.arbor as sim
from pyNN.morphology import NeuroMLMorphology


sim.setup()

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ArborSingleCellExample"
cell_class.ion_channels = {'na': sim.NaChannel, 'kdr': sim.KdrChannel}

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
    na={"conductance_density": 0.120},
    kdr={"conductance_density": 0.036}
)

cells = sim.Population(1, cell_type, initial_values={"v": -40})

stim = sim.DCSource(start=10, stop=12, amplitude=0.8)
stim.inject_into(cells, location="soma")

cells.record("v", locations={"soma": "soma"})


# (4) Define a recipe for a single cell and set of probes upon it.
# This constitutes the corresponding generic recipe version of
# `single_cell_model.py`.


class single_recipe(arbor.recipe):
    # (4.1) The base class constructor must be called first, to ensure that
    # all memory in the wrapped C++ class is initialized correctly.
    def __init__(self):
        arbor.recipe.__init__(self)
        #self.the_props = arbor.neuron_cable_properties()
        #breakpoint()

    # (4.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (4.3) Override the cell_kind method
    def cell_kind(self, gid):
        return cells.arbor_cell_kind

    # (4.4) Override the cell_description method
    def cell_description(self, gid):
        return cells.arbor_cell_description(gid)

    # (4.5) Override the probes method with a voltage probe located on "midpoint"
    def probes(self, gid):
        return cells.recorder._get_arbor_probes(0)

    # (4.6) Override the global_properties method
    def global_properties(self, kind):
        #return self.the_props
        return arbor.neuron_cable_properties()


# (5) Instantiate recipe.

recipe = single_recipe()

# (6) Create simulation. When their defaults are sufficient, context and domain decomposition don't
# have to be manually specified and the simulation can be created with just the recipe as argument.

sim = arbor.simulation(recipe)

# (7) Create and run simulation and set up 10 kHz (every 0.1 ms) sampling on the probe.
# The probe is located on cell 0, and is the 0th probe on that cell, thus has probeset_id (0, 0).

sim.record(arbor.spike_recording.all)
handle = sim.sample((0, 0), arbor.regular_schedule(0.1))
sim.run(tfinal=30)

# (8) Collect results.

spikes = sim.spikes()
data, meta = sim.samples(handle)[0]

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes["time"]:
        print("{:3.3f}".format(t))
else:
    print("no spikes")

print("Plotting results ...")

original_data = np.loadtxt("arbor_single_cell_recipe.original_data")
plt.plot(original_data[:, 0], original_data[:, 1], "b-", lw=3)
plt.plot(data[:, 0], data[:, 1], 'g-')
plt.xlabel("t (ms)")
plt.ylabel("v (mV)")
plt.savefig("arbor_single_cell_PyNN_step3.png")
