"""
System tests for multicompartment (MC) neurons.

These run on the backends with MC support (Arbor and NEURON) and fill gaps left
by test_scenario5 (which records spikes only): recording analog signals at named
locations and from density-mechanism state variables, and sub-threshold current
injection. Several assertions are regression guards for bugs found while
supporting Arbor 0.10 - 0.12 (spike recording returning nothing, and
probe-location label resolution).
"""

try:
    import morphio  # noqa: F401
    have_morphio = True
except ImportError:
    have_morphio = False

try:
    from neuroml import Morphology, Segment, Point3DWithDiam as P
    have_neuroml = True
except ImportError:
    have_neuroml = False

from pyNN.morphology import NeuroMLMorphology
from pyNN.parameters import IonicSpecies

import pytest

from .fixtures import run_with_simulators


def _build_mc_cell_type(sim):
    """A two-compartment (soma + dendrite) multicompartment cell type."""
    soma = Segment(proximal=P(x=18.8, y=0, z=0, diameter=18.8),
                   distal=P(x=0, y=0, z=0, diameter=18.8), name="soma", id=0)
    dend = Segment(proximal=P(x=0, y=0, z=0, diameter=2),
                   distal=P(x=-500, y=0, z=0, diameter=2), name="dendrite",
                   parent=soma, id=1)
    cell_class = sim.MultiCompartmentNeuron
    cell_class.label = "MCTestNeuron"
    cell_class.ion_channels = {'pas': sim.PassiveLeak, 'na': sim.NaChannel,
                               'kdr': sim.KdrChannel}
    return cell_class(
        morphology=NeuroMLMorphology(Morphology(segments=(soma, dend))),
        cm=1.0, Ra=500.0,
        ionic_species={"na": IonicSpecies("na", reversal_potential=50.0),
                       "k": IonicSpecies("k", reversal_potential=-77.0)},
        pas={"conductance_density": sim.morphology.uniform('all', 0.0003),
             "e_rev": -54.3},
        na={"conductance_density": sim.morphology.uniform('soma', 0.120)},
        kdr={"conductance_density": sim.morphology.uniform('soma', 0.036)},
    )


@run_with_simulators("arbor", "neuron")
def test_mc_recording_at_locations(sim):
    """Record v at two locations, density-mechanism states, and spikes.

    Guards: probe-tag matching, probe-location label resolution, and spike
    recording actually returning data (all had silent bugs across Arbor versions).
    """
    if not have_morphio:
        pytest.skip("morphio not available")
    if not have_neuroml:
        pytest.skip("libNeuroML not available")

    sim.setup(timestep=0.025)
    cell_type = _build_mc_cell_type(sim)
    cells = sim.Population(2, cell_type, initial_values={'v': -60.0})

    # inject a supra-threshold current into cell 0 only
    step = sim.DCSource(amplitude=0.5, start=20.0, stop=180.0)
    step.inject_into(cells[0:1], location="soma")

    cells.record('spikes')
    cells.record(['na.m', 'na.h', 'kdr.n'], locations="soma")
    cells.record('v', locations=("soma", "dendrite"))

    sim.run(200.0)
    data = cells.get_data().segments[0]

    # analog signals exist at both locations, one column per cell
    soma_v = data.filter(name='soma.v')[0]
    dend_v = data.filter(name='dendrite.v')[0]
    assert soma_v.shape[1] == 2
    assert dend_v.shape[1] == 2
    assert soma_v.shape[0] > 1

    soma_v = soma_v.magnitude
    # injected cell 0 fires (crosses 0 mV); uninjected cell 1 stays sub-threshold
    assert soma_v[:, 0].max() > 0.0
    assert soma_v[:, 1].max() < 0.0

    # density-mechanism state variables are recorded and physically plausible (0..1)
    for name in ('soma.na.m', 'soma.na.h', 'soma.kdr.n'):
        sig = data.filter(name=name)[0]
        assert sig.shape[1] == 2
        assert sig.magnitude.min() >= -1e-6
        assert sig.magnitude.max() <= 1.0 + 1e-6

    # spikes are recorded: injected cell fires, uninjected cell is silent
    spiketrains = data.spiketrains
    assert len(spiketrains) == 2
    counts = sorted(len(st) for st in spiketrains)
    assert counts[0] == 0
    assert counts[1] >= 1

    sim.end()


@run_with_simulators("arbor", "neuron")
def test_mc_subthreshold_dc(sim):
    """A small current depolarises the cell without triggering a spike."""
    if not have_morphio:
        pytest.skip("morphio not available")
    if not have_neuroml:
        pytest.skip("libNeuroML not available")

    sim.setup(timestep=0.01)
    cell_type = _build_mc_cell_type(sim)
    cells = sim.Population(1, cell_type, initial_values={'v': -60.0})
    step = sim.DCSource(amplitude=0.04, start=50.0, stop=150.0)
    step.inject_into(cells[0:1], location="soma")
    cells.record('v', locations=("soma",))
    cells.record('spikes')

    sim.run(200.0)
    data = cells.get_data().segments[0]
    soma_v = data.filter(name='soma.v')[0].magnitude[:, 0]

    assert len(data.spiketrains[0]) == 0            # no spike
    assert soma_v.max() < -40.0                     # stayed sub-threshold
    assert soma_v.max() > -60.0                     # but did depolarise

    sim.end()
