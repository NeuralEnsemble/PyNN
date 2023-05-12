#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import numpy as np
import matplotlib.pyplot as plt
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
import pyNN.arbor as sim
from pyNN.morphology import NeuroMLMorphology
from pyNN.arbor.cells import convert_point, region_name_to_tag

# The corresponding generic recipe version of `single_cell_model.py`.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm

soma = Segment(proximal=P(x=-3, y=0, z=0, diameter=6),
               distal=P(x=3, y=0, z=0, diameter=6),
               name="soma", id=0)

morphology = NeuroMLMorphology(NMLMorphology(segments=(soma,)))

tree = arbor.segment_tree()
for i, segment in enumerate(morphology.segments):
    assert i == segment.id
    prox = convert_point(segment.proximal)
    dist = convert_point(segment.distal)
    tag = region_name_to_tag(segment.name)
    if segment.parent is None:
        parent = arbor.mnpos
    else:
        parent = segment.parent.id
    print(i, parent, prox, dist, tag)
    tree.append(parent, prox, dist, tag=tag)

#tree = arbor.segment_tree()
#tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint

labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties

decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("hh"))
    .place('"midpoint"', arbor.iclamp(10, 2, 0.8), "iclamp")
    .place('"midpoint"', arbor.threshold_detector(-10), "detector")
)

cell = arbor.cable_cell(tree, decor, labels)

# (4) Define a recipe for a single cell and set of probes upon it.
# This constitutes the corresponding generic recipe version of
# `single_cell_model.py`.


class single_recipe(arbor.recipe):
    # (4.1) The base class constructor must be called first, to ensure that
    # all memory in the wrapped C++ class is initialized correctly.
    def __init__(self):
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()

    # (4.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (4.3) Override the cell_kind method
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (4.4) Override the cell_description method
    def cell_description(self, gid):
        return cell

    # (4.5) Override the probes method with a voltage probe located on "midpoint"
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"midpoint"')]

    # (4.6) Override the global_properties method
    def global_properties(self, kind):
        return self.the_props


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
plt.savefig("arbor_single_cell_PyNN_step1.png")
