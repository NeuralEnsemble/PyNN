
import matplotlib.pyplot as plt
from neuroml import Morphology as NMLMorphology, Segment, Point3DWithDiam as P
from pyNN.morphology import NeuroMLMorphology, uniform
from pyNN.utility import get_simulator

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

sim.setup(timestep=args["dt"])

cell_class = sim.MultiCompartmentNeuron
cell_class.label = "ArborSimpleDendriteExample"
cell_class.ion_channels = {'pas': sim.PassiveLeak}

n_segments = round(args["length"] / args["cv_policy_max_extent"])
seg_length = args["cv_policy_max_extent"]
diam = 2 * args["radius"]
cable = [
    Segment(proximal=P(x=0, y=0, z=0, diameter=diam),
            distal=P(x=0, y=0, z=seg_length, diameter=diam),
            name="seg0", id=0)
]
for i in range(1, n_segments):
    cable.append(
        Segment(proximal=P(x=0, y=0, z=seg_length * i, diameter=diam),
                distal=P(x=0, y=0, z=seg_length * (i + 1), diameter=diam),
                name=f"seg{i}", id=i, parent=cable[i - 1])
    )

morphology = NeuroMLMorphology(NMLMorphology(segments=cable))

cell_type = cell_class(
    morphology=morphology,
    ionic_species={},
    cm=args["cm"] * 100,
    Ra=args["rL"],
    pas={
        "conductance_density": uniform('all', args["g"]),
        "e_rev": args["Vm"]
    }
)

initial_values = {
    "v": args["Vm"],
}
dendrite = sim.Population(1, cell_type, initial_values=initial_values)

stim = sim.DCSource(
    start=args["stimulus_start"],
    stop=args["stimulus_start"] + args["stimulus_duration"],
    amplitude=args["stimulus_amplitude"]
)
stim.inject_into(dendrite, location="seg0")

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
    plt.savefig(f"{options.simulator}_single_cell_cable_PyNN.png")
