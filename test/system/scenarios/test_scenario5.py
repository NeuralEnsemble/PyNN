import sys
import numpy as np

try:
    from neuroml import Morphology, Segment, Point3DWithDiam as P
    have_neuroml = True
except ImportError:
    have_neuroml = False

from pyNN.utility import init_logging
from pyNN.morphology import NeuroMLMorphology
from pyNN.parameters import IonicSpecies

import pytest

from .fixtures import run_with_simulators


@run_with_simulators("arbor", "neuron")
def test_scenario5(sim):
    """
    Array of multi-compartment neurons, each injected with a different current.
    """
    if not have_neuroml:
        pytest.skip("libNeuroML not available")

    init_logging(logfile=None, debug=True)

    sim.setup(timestep=0.01)

    soma = Segment(proximal=P(x=18.8, y=0, z=0, diameter=18.8),
                distal=P(x=0, y=0, z=0, diameter=18.8),
                name="soma", id=0)
    dend = Segment(proximal=P(x=0, y=0, z=0, diameter=2),
                distal=P(x=-500, y=0, z=0, diameter=2),
                name="dendrite",
                parent=soma, id=1)

    cell_class = sim.MultiCompartmentNeuron
    cell_class.label = "ExampleMultiCompartmentNeuron"
    cell_class.ion_channels = {'pas': sim.PassiveLeak, 'na': sim.NaChannel, 'kdr': sim.KdrChannel}

    cell_type = cell_class(
        morphology=NeuroMLMorphology(Morphology(segments=(soma, dend))),
        cm=1.0,    # mF / cm**2
        Ra=500.0,  # ohm.cm
        ionic_species={
                "na": IonicSpecies("na", reversal_potential=50.0),
                "k": IonicSpecies("k", reversal_potential=-77.0)
        },
        pas={"conductance_density": sim.morphology.uniform('all', 0.0003),
             "e_rev":-54.3},
        na={"conductance_density": sim.morphology.uniform('soma', 0.120)},
        kdr={"conductance_density": sim.morphology.uniform('soma', 0.036)}
    )

    neurons = sim.Population(5, cell_type, initial_values={'v': -60.0})

    I = (0.04, 0.11, 0.13, 0.15, 0.18)
    currents = [sim.DCSource(start=50, stop=150, amplitude=amp)
                for amp in I]
    for j, (neuron, current) in enumerate(zip(neurons, currents)):
        if j % 2 == 0:                              # these should
            neuron.inject(current, location="soma") # be entirely
        else:                                       # equivalent
            current.inject_into([neuron], location="soma")

    neurons.record('spikes')

    sim.run(200.0)

    spiketrains = neurons.get_data().segments[0].spiketrains
    assert len(spiketrains) == 5
    assert len(spiketrains[0]) == 0  # first cell does not fire
    # expected values taken from the average of simulations with NEURON and Arbor
    expected_spike_times = [
        np.array([]),
        np.array([52.41]),
        np.array([52.15, 68.45, 84.73, 101.02, 117.31, 133.61, 149.9]),
        np.array([51.96, 67.14, 82.13, 97.11, 112.08, 127.06, 142.04]),
        np.array([51.75, 65.86, 79.7, 93.51, 107.33, 121.14, 134.96, 148.77])
    ]
    spike_times = [np.array(st) for st in spiketrains[1:]]
    max_error = 0
    for a, b in zip(spike_times, expected_spike_times[1:]):
        if a.size == b.size:
            max_error += abs((a - b) / b).max()
        else:
            max_error += 1
    print("max error =", max_error)
    assert max_error < 0.005, max_error
    sim.end()
    if "pytest" not in sys.modules:
        return a, b, spike_times


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_scenario5(sim)
