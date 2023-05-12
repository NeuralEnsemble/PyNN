
import matplotlib.pyplot as plt
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.morphology import NeuroMLMorphology, uniform
from pyNN.utility import get_simulator
from pyNN.arbor.cells import NativeCellType

from arbor_single_cell_cable_recipe_with_segments import Cable

sim, options = get_simulator()

args = {
    "Vm": -65,
    "length": 1000,
    "radius": 1,
    "cm": 0.01,
    "rL": 90,
    "g": 0.001,
    "stimulus_start": 10,
    "stimulus_duration": 0.1,
    "stimulus_amplitude": 1.0,
    "cv_policy_max_extent": 10,
    "dt": 0.001
}

sim.setup(timestep=args.pop("dt"))

recipe = Cable(probes=None, **args)

(tree, decor, labels) = recipe._cell_description(0)
cell_type = NativeCellType(tree=tree, decor=lambda i: decor, labels=labels)

initial_values = {
    "v": args["Vm"],
}
dendrite = sim.Population(1, cell_type, initial_values=initial_values)

recording_locations = {
    "seg0": "seg0",
    "seg10": "seg10",
    "seg20": "seg20",
    "seg30": "seg30",
    "seg40": "seg40",
    "seg50": "seg50",
    "seg60": "seg60",
    "seg70": "seg70",
    "seg80": "seg80",
    "seg90": "seg90",
    "seg99": "seg99",
}
dendrite.record("v", locations=recording_locations)


sim.run(30)

data = dendrite.get_data().segments[0]

for sig in sorted(data.analogsignals, key=lambda s: s.name):
    plt.plot(sig.times, sig, label=sig.name)
    plt.xlim(9, 14)
    plt.legend()
    plt.savefig("arbor_single_cell_cable_PyNN_native.png")
