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


def _build_mc_cell_type(sim, synapse_locations=None):
    """A two-compartment (soma + dendrite) multicompartment cell type."""
    soma = Segment(proximal=P(x=18.8, y=0, z=0, diameter=18.8),
                   distal=P(x=0, y=0, z=0, diameter=18.8), name="soma", id=0)
    dend = Segment(proximal=P(x=0, y=0, z=0, diameter=2),
                   distal=P(x=-500, y=0, z=0, diameter=2), name="dendrite",
                   parent=soma, id=1)
    synapses = {}
    post_synaptic_entities = {}
    if synapse_locations is not None:
        post_synaptic_entities = {'AMPA': sim.CondExpPostSynapticResponse}
        synapses["AMPA"] = {"locations": synapse_locations, "e_syn": 0.0, "tau_syn": 2.0}
    # subclass, rather than modify sim.MultiCompartmentNeuron, which other tests use
    cell_class = type("MCTestNeuron", (sim.MultiCompartmentNeuron,), {
        "label": "MCTestNeuron",
        "ion_channels": {'pas': sim.PassiveLeak, 'na': sim.NaChannel, 'kdr': sim.KdrChannel},
        "post_synaptic_entities": post_synaptic_entities,
    })
    return cell_class(
        morphology=NeuroMLMorphology(Morphology(segments=(soma, dend))),
        cm=1.0, Ra=500.0,
        ionic_species={"na": IonicSpecies("na", reversal_potential=50.0),
                       "k": IonicSpecies("k", reversal_potential=-77.0)},
        pas={"conductance_density": sim.morphology.uniform('all', 0.0003),
             "e_rev": -54.3},
        na={"conductance_density": sim.morphology.uniform('soma', 0.120)},
        kdr={"conductance_density": sim.morphology.uniform('soma', 0.036)},
        **synapses
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


@run_with_simulators("arbor", "neuron")
def test_mc_recording_at_distances(sim):
    """Record v at path distances from the root of the morphology.

    The soma is 18.8 µm long, so 100 µm and 400 µm are in the dendrite.
    A depolarisation injected at the soma attenuates along the dendrite.
    """
    if not have_morphio:
        pytest.skip("morphio not available")
    if not have_neuroml:
        pytest.skip("libNeuroML not available")

    sim.setup(timestep=0.01)
    cell_type = _build_mc_cell_type(sim)
    cells = sim.Population(1, cell_type, initial_values={'v': -60.0})
    step = sim.DCSource(amplitude=0.04, start=50.0, stop=150.0)
    step.inject_into(cells, location="soma")
    cells.record('v', locations=["soma", sim.morphology.at_distances("dendrite", [100, 400])])

    sim.run(150.0)
    data = cells.get_data().segments[0]
    sim.end()

    assert set(sig.name for sig in data.analogsignals) == {
        "soma.v", "dendrite.100um.v", "dendrite.400um.v"}
    depolarisation = {}
    for name in ("soma.v", "dendrite.100um.v", "dendrite.400um.v"):
        v = data.filter(name=name)[0].magnitude[:, 0]
        depolarisation[name] = v[-1] - v[4999]  # at 150 ms relative to 50 ms
    assert (depolarisation["soma.v"] > depolarisation["dendrite.100um.v"]
            > depolarisation["dendrite.400um.v"] > 0)


@run_with_simulators("arbor", "neuron")
def test_mc_synapse_at_distance(sim):
    """A synapse placed with at_distances() gives a larger EPSP at its own location."""
    if not have_morphio:
        pytest.skip("morphio not available")
    if not have_neuroml:
        pytest.skip("libNeuroML not available")

    sim.setup(timestep=0.025)
    cell_type = _build_mc_cell_type(
        sim, synapse_locations=sim.morphology.at_distances("dendrite", [450]))
    cells = sim.Population(1, cell_type, initial_values={'v': -60.0})
    source = sim.Population(1, sim.SpikeSourceArray(spike_times=[20.0]))
    sim.Projection(source, cells, sim.AllToAllConnector(),
                   sim.StaticSynapse(weight=0.005, delay=1.0), receptor_type="AMPA")
    cells.record('v', locations=sim.morphology.at_distances("dendrite", [100, 450]))

    sim.run(50.0)
    data = cells.get_data().segments[0]
    sim.end()

    epsp = {}
    for name in ("dendrite.100um.v", "dendrite.450um.v"):
        v = data.filter(name=name)[0].magnitude[:, 0]
        epsp[name] = v.max() - v[700]  # relative to v at 17.5 ms
    assert epsp["dendrite.450um.v"] > epsp["dendrite.100um.v"] > 0


@run_with_simulators("arbor", "neuron")
def test_mc_at_distances_on_branches(sim):
    """A distance reached on more than one branch gives a location on each branch.

    The dendrite has a 200 µm trunk and two branches, of length 141 µm and 224 µm.
    """
    if not have_neuroml:
        pytest.skip("libNeuroML not available")

    sim.setup(timestep=0.025)
    soma = Segment(proximal=P(x=18.8, y=0, z=0, diameter=18.8),
                   distal=P(x=0, y=0, z=0, diameter=18.8), name="soma", id=0)
    trunk = Segment(proximal=P(x=0, y=0, z=0, diameter=2),
                    distal=P(x=-200, y=0, z=0, diameter=2), name="dendrite", parent=soma, id=1)
    branch1 = Segment(proximal=P(x=-200, y=0, z=0, diameter=1),
                      distal=P(x=-300, y=100, z=0, diameter=1), name="dendrite", parent=trunk, id=2)
    branch2 = Segment(proximal=P(x=-200, y=0, z=0, diameter=1),
                      distal=P(x=-400, y=-100, z=0, diameter=1), name="dendrite", parent=trunk, id=3)
    cell_class = type("BranchedNeuron", (sim.MultiCompartmentNeuron,), {
        "label": "BranchedNeuron",
        "ion_channels": {'pas': sim.PassiveLeak},
        "post_synaptic_entities": {},
    })
    cell_type = cell_class(
        morphology=NeuroMLMorphology(Morphology(segments=(soma, trunk, branch1, branch2))),
        cm=1.0, Ra=100.0, ionic_species={},
        pas={"conductance_density": sim.morphology.uniform('all', 0.0003), "e_rev": -65.0})
    cells = sim.Population(1, cell_type, initial_values={'v': -65.0})
    cells.record('v', locations=sim.morphology.at_distances("dendrite", [100, 300, 400, 500]))

    sim.run(1.0)
    data = cells.get_data().segments[0]
    sim.end()

    assert set(sig.name for sig in data.analogsignals) == {
        "dendrite.100um.v", "dendrite.300um#0.v", "dendrite.300um#1.v", "dendrite.400um.v"}
