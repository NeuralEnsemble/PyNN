#!/usr/bin/env python3

import arbor
import numpy as np
import matplotlib.pyplot as plt

# The corresponding generic recipe version of `single_cell_model.py`.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm

tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint

labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties

def build_decor(v_init, start, duration, amplitude):
    return (
        arbor.decor()
        .set_property(Vm=v_init)
        .paint('"soma"', arbor.density("hh"))
        .place('"midpoint"', arbor.iclamp(start, duration, amplitude), "iclamp")
        .place('"midpoint"', arbor.threshold_detector(-10), "detector")
    )

cells = [
    arbor.cable_cell(tree, build_decor(-60, 10, 2, 0.01), labels),
    arbor.cable_cell(tree, build_decor(-50, 8, 6, 0.003), labels)
]

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
        return 2

    # (4.3) Override the cell_kind method
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (4.4) Override the cell_description method
    def cell_description(self, gid):
        return cells[gid]

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
handles = [
    sim.sample((0, 0), arbor.regular_schedule(0.1)),
    sim.sample((1, 0), arbor.regular_schedule(0.1))
]

sim.run(tfinal=30)

# (8) Collect results.

spikes = sim.spikes()
all_data = []
for handle in handles:
    data, meta = sim.samples(handle)[0]
    all_data.append(data)

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes["time"]:
        print("{:3.3f}".format(t))
else:
    print("no spikes")

print("Plotting results ...")

plt.plot(all_data[0][:, 0], all_data[0][:, 1])
plt.plot(all_data[1][:, 0], all_data[1][:, 1])
plt.xlabel("t (ms)")
plt.ylabel("v (mV)")
plt.savefig("arbor_two_cells_recipe.png")

#np.savetxt("arbor_two_cells_recipe.original_data", data)
