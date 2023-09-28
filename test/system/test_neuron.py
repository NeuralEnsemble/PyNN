import os
from numpy.testing import assert_array_equal
from pyNN.random import RandomDistribution
from pyNN.parameters import IonicSpecies
from pyNN.space import Grid2D, RandomStructure, Sphere
from pyNN.utility import init_logging
import quantities as pq
import numpy as np
import pytest


try:
    import pyNN.neuron
    from pyNN.neuron.cells import _new_property, NativeCellType
    from nrnutils import Mechanism, Section, DISTAL
    have_neuron = True
except ImportError:
    have_neuron = False
import pytest


try:
    from neuroml import Morphology, Segment, Point3DWithDiam as P
    from pyNN.morphology import NeuroMLMorphology, load_morphology
    import neuroml.loaders
    have_neuroml = True
except ImportError:
    have_neuroml = False


skip_ci = False
if "JENKINS_SKIP_TESTS" in os.environ:
    skip_ci = os.environ["JENKINS_SKIP_TESTS"] == "1"


def test_ticket168():
    """
    Error setting firing rate of `SpikeSourcePoisson` after `reset()` in NEURON
    http://neuralensemble.org/trac/PyNN/ticket/168
    """
    if not have_neuron:
        pytest.skip("neuron not available")
    pynn = pyNN.neuron
    pynn.setup()
    cell = pynn.Population(1, pynn.SpikeSourcePoisson(), label="cell")
    cell[0].rate = 12
    pynn.run(10.)
    pynn.reset()
    cell[0].rate = 12
    pynn.run(10.)
    assert pynn.get_current_time() == pytest.approx(10.0)  # places=11)
    assert cell[0]._cell.interval == 1000.0 / 12.0


class SimpleNeuron(object):

    def __init__(self, **parameters):
        # define ion channel parameters
        leak = Mechanism('pas', e=-65, g=parameters['g_leak'])
        hh = Mechanism('hh', gl=parameters['g_leak'], el=-65,
                       gnabar=parameters['gnabar'], gkbar=parameters['gkbar'])
        # create cable sections
        self.soma = Section(L=30, diam=30, mechanisms=[hh])
        self.apical = Section(L=600, diam=2, nseg=5, mechanisms=[leak], parent=self.soma,
                              connection_point=DISTAL)
        self.basilar = Section(L=600, diam=2, nseg=5, mechanisms=[leak], parent=self.soma)
        self.axon = Section(L=1000, diam=1, nseg=37, mechanisms=[hh])
        self.section_labels = {
            "soma": self.soma,
            "apical": self.apical,
            "basilar": self.basilar,
            "axon": self.axon
        }
        # synaptic input
        self.apical.add_synapse('ampa', 'Exp2Syn', e=0.0, tau1=0.1, tau2=5.0)

        # needed for PyNN
        self.source_section = self.soma
        self.source = self.soma(0.5)._ref_v
        self.parameter_names = ('g_leak', 'gnabar', 'gkbar')
        self.traces = defaultdict(list)
        self.recording_time = False

    def _set_g_leak(self, value):
        for sec in (self.apical, self.basilar):
            for seg in sec:
                seg.pas.g = value
        for sec in (self.soma, self.axon):
            for seg in sec:
                seg.hh.gl = value

    def _get_g_leak(self):
        return self.apical(0.5).pas.g
    g_leak = property(fget=_get_g_leak, fset=_set_g_leak)

    def _set_gnabar(self, value):
        for sec in (self.soma, self.axon):
            for seg in sec:
                seg.hh.gnabar = value

    def _get_gnabar(self):
        return self.soma(0.5).hh.gnabar
    gnabar = property(fget=_get_gnabar, fset=_set_gnabar)

    def _set_gkbar(self, value):
        for sec in (self.soma, self.axon):
            for seg in sec:
                seg.hh.gkbar = value

    def _get_gkbar(self):
        return self.soma(0.5).hh.gkbar
    gkbar = property(fget=_get_gkbar, fset=_set_gkbar)

    def memb_init(self):
        """needed for PyNN"""
        for sec in (self.soma, self.axon, self.apical, self.basilar):
            for seg in sec:
                seg.v = self.initial_values["v"]


if have_neuron:
    class SimpleNeuronType(NativeCellType):
        default_parameters = {'g_leak': 0.0002, 'gkbar': 0.036, 'gnabar': 0.12}
        default_initial_values = {'v': -65.0}
        recordable = ['v', 'hh.ina']  # this is not good - over-ride Population.can_record()?
        units = {'v': 'mV', 'hh.ina': 'mA/cm**2'}
        receptor_types = ['apical.ampa']
        model = SimpleNeuron


def test_electrical_synapse():
    pytest.skip("Skipping test for now as it produces a segmentation fault")
    if skip_ci:
        pytest.skip("Skipping test on CI server as it produces a segmentation fault")
    p1 = pyNN.neuron.Population(4, pyNN.neuron.standardmodels.cells.HH_cond_exp())
    p2 = pyNN.neuron.Population(4, pyNN.neuron.standardmodels.cells.HH_cond_exp())
    syn = pyNN.neuron.ElectricalSynapse(weight=1.0)
    C = pyNN.connectors.FromListConnector(np.array([[0, 0, 1.0],
                                                       [0, 1, 1.0],
                                                       [2, 2, 1.0],
                                                       [3, 2, 1.0]]),
                                          column_names=['weight'])
    prj = pyNN.neuron.Projection(p1, p2, C, syn,
                                 source='source_section.gap', receptor_type='source_section.gap')
    current_source = pyNN.neuron.StepCurrentSource(amplitudes=[1.0], times=[100])
    p1[0:1].inject(current_source)
    p2[2:3].inject(current_source)
    p1.record('v')
    p2.record('v')
    pyNN.neuron.run(200)
    p1_trace = p1.get_data(('v',)).segments[0].analogsignals[0]
    p2_trace = p2.get_data(('v',)).segments[0].analogsignals[0]
    # Check the local forward connection
    assert p2_trace[:, 0].max() - p2_trace[:, 0].min() > 50
    # Check the remote forward connection
    assert p2_trace[:, 1].max() - p2_trace[:, 1].min() > 50
    # Check the local backward connection
    assert p1_trace[:, 2].max() - p2_trace[:, 2].min() > 50
    # Check the remote backward connection
    assert p1_trace[:, 3].max() - p2_trace[:, 3].min() > 50


def test_record_native_model():
    pytest.skip("to fix once mc branch is stable")
    if not have_neuron:
        pytest.skip("neuron not available")
    nrn = pyNN.neuron

    init_logging(logfile=None, debug=True)
    nrn.setup()

    parameters = {'g_leak': 0.0003}
    p1 = nrn.Population(10, SimpleNeuronType(**parameters))
    print(p1.get('g_leak'))
    p1.rset('gnabar', RandomDistribution('uniform', low=0.10, high=0.14))
    print(p1.get('gnabar'))
    p1.initialize(v=-63.0)

    current_source = nrn.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                                           amplitudes=[0.4, 0.6, -0.2, 0.2])
    p1.inject(current_source)

    p2 = nrn.Population(1, nrn.SpikeSourcePoisson(rate=100.0))

    p1.record('v', locations={'apical': 'apical'})
    p1.record('hh.ina', locations={'soma': 'soma'})

    connector = nrn.AllToAllConnector()
    syn = nrn.StaticSynapse(weight=0.1)
    prj_alpha = nrn.Projection(p2, p1, connector, syn, receptor_type='apical.ampa')

    nrn.run(250.0)

    data = p1.get_data().segments[0].analogsignals
    assert len(data) == 2  # one array per variable
    names = set(sig.name for sig in data)
    assert names == set(('apical.v', 'soma.ina'))
    apical_v = [sig for sig in data if sig.name == 'apical.v'][0]
    soma_i = [sig for sig in data if sig.name == 'soma.ina'][0]
    assert apical_v.sampling_rate == 10.0 * pq.kHz
    assert apical_v.units == pq.mV
    assert soma_i.units == pq.mA / pq.cm**2
    assert apical_v.t_start == 0.0 * pq.ms
    # would prefer if it were 250.0, but this is a fundamental Neo issue
    assert apical_v.t_stop == 250.1 * pq.ms
    assert apical_v.shape == (2501, 10)
    return data


def test_tsodyks_markram_synapse():
    if not have_neuron:
        pytest.skip("neuron not available")
    sim = pyNN.neuron
    sim.setup()
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(10, 100, 10)))
    neurons = sim.Population(5, sim.IF_cond_exp(
        e_rev_I=-75, tau_syn_I=np.arange(0.2, 0.7, 0.1)))
    synapse_type = sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0,
                                             tau_facil=1000.0, weight=0.01,
                                             delay=0.5)
    connector = sim.AllToAllConnector()
    prj = sim.Projection(spike_source, neurons, connector,
                         receptor_type='inhibitory',
                         synapse_type=synapse_type)
    neurons.record('gsyn_inh')
    sim.run(100.0)
    tau_psc = np.array([c.weight_adjuster.tau_syn for c in prj.connections])
    assert_array_equal(tau_psc, np.arange(0.2, 0.7, 0.1))


def test_artificial_cells():
    if not have_neuron:
        pytest.skip("neuron not available")
    sim = pyNN.neuron
    sim.setup()
    input = sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(10, 100, 10)))
    p1 = sim.Population(3, sim.IntFire1(tau=10, refrac=2))
    p2 = sim.Population(3, sim.IntFire2())
    p3 = sim.Population(3, sim.IntFire4())
    projections = []
    for p in (p1, p2, p3):
        projections.append(
            sim.Projection(input, p, sim.AllToAllConnector(), sim.StaticSynapse(weight=0.1, delay=0.5),
                           receptor_type="default")
        )
        p.record('m')
    sim.run(100.0)


def test_2_compartment():
    if not (have_neuron and have_neuroml):
        pytest.skip("Need neuron and neuroml")
    sim = pyNN.neuron

    sim.setup(timestep=0.025)
    soma = Segment(proximal=P(x=0, y=0, z=0, diameter=18.8),
                   distal=P(x=18.8, y=0, z=0, diameter=18.8),
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
        cm=1.01,
        Ra=500.0,
        ionic_species={
                "na": IonicSpecies("na", reversal_potential=50.1),
                "k": IonicSpecies("k", reversal_potential=-77.7)
        },
        pas={"conductance_density": sim.morphology.uniform('all', 0.00033),
                "e_rev":-54.32},
        na={"conductance_density": sim.morphology.uniform('soma', 0.121)},
        kdr={"conductance_density": sim.morphology.uniform('soma', 0.0363)}
    )

    cells = sim.Population(2, cell_type, initial_values={'v': [-60.0, -70.0]})  #*mV})
    step_current = sim.DCSource(amplitude=0.1, start=50.0, stop=150.0)
    step_current.inject_into(cells[0:1], location="soma")
    step_current.inject_into(cells[1:2], location="dendrite")

    cells.record('spikes')
    cells.record(['na.m', 'na.h', 'kdr.n'], locations="soma")
    cells.record('v', locations=["soma", "dendrite"])

    sim.run(200.0)

    data = cells.get_data().segments[0]

    hcell0 = cells[0]._cell
    soma_id,  = hcell0.section_labels["soma"]
    hsoma = hcell0.sections[soma_id]
    assert abs(hsoma.L - 18.8) < 1e-6
    assert abs(hsoma.diam - 18.8) < 1e-6
    assert hsoma.cm == 1.01
    assert hsoma.gnabar_hh == 0.121
    assert hsoma.gkbar_hh == 0.0363
    assert hsoma.gl_hh == 0.0
    assert hsoma.ena == 50.1
    assert hsoma.ek == -77.7
    assert hsoma.e_pas == -54.32
    assert hsoma.g_pas == 0.00033

    dend_id, = hcell0.section_labels["dendrite"]
    hdend = hcell0.sections[dend_id]
    assert hdend.L == 500.0
    assert hdend.diam == 2.0
    assert hsoma.cm == 1.01
    assert hdend.e_pas == -54.32
    assert hdend.g_pas == 0.00033
    assert not hasattr(hdend, "ena")

    vm_soma = data.filter(name="soma.v")[0]
    assert vm_soma[0, 0] == pq.Quantity(-60.0, "mV")
    assert vm_soma[0, 1] == pq.Quantity(-70.0, "mV")
    vm_dend = data.filter(name="dendrite.v")[0]
    assert vm_dend[0, 0] == pq.Quantity(-60.0, "mV")
    assert vm_dend[0, 1] == pq.Quantity(-70.0, "mV")


def test_mc_network():
    if not (have_neuron and have_neuroml):
        pytest.skip("Need neuron and neuroml packages")
    sim = pyNN.neuron
    from pyNN.neuron.morphology import (uniform, by_distance, random_placement as rp, centre, soma, apical_dendrites, dendrites, random_section)

    sim.setup()

    try:
        pyr_morph = load_morphology(
            "http://neuromorpho.org/dableFiles/kisvarday/CNG%20version/oi15rpy4-1.CNG.swc",
            replace_axon=None)
    except Exception as err:
        pytest.skip(f"Problem downloading morphology file: {str(err)}")

    pyramidal_cell_class = sim.MultiCompartmentNeuron
    pyramidal_cell_class.label = "PyramidalNeuron"
    pyramidal_cell_class.ion_channels = {
        'pas': sim.PassiveLeak,
        'na': sim.NaChannel,
        'kdr': sim.KdrChannel
    }
    pyramidal_cell_class.post_synaptic_entities = {'AMPA': sim.CondExpPostSynapticResponse,
                                                   'GABA_A': sim.CondExpPostSynapticResponse}

    pyramidal_cell = pyramidal_cell_class(
                        morphology=pyr_morph,
                        pas={"conductance_density": uniform('all', 0.0003)},
                        na={"conductance_density": uniform('soma', 0.120)},
                        kdr={"conductance_density": by_distance(apical_dendrites(), lambda d: 0.05*d/200.0)},
                        ionic_species={
                            "na": IonicSpecies("na", reversal_potential=50.0),
                            "k": IonicSpecies("k", reversal_potential=-77.0)
                        },
                        cm=1.0,
                        Ra=500.0,
                        AMPA={"locations": rp(uniform('all', 0.05)),  # number per µm
                              "e_syn": 0.0,
                              "tau_syn": 2.0},
                        GABA_A={"locations": rp(by_distance(dendrites(), lambda d: 0.05 * (d < 50.0))),  # number per µm
                                "e_syn": -70.0,
                                "tau_syn": 5.0})

    pyramidal_cells = sim.Population(2, pyramidal_cell, initial_values={'v': -60.0}, structure=Grid2D())
    inputs = sim.Population(1000, sim.SpikeSourcePoisson(rate=1000.0))

    pyramidal_cells.record('spikes')
    pyramidal_cells[:1].record('v', locations=centre(soma()))
    pyramidal_cells[:1].record('v', locations=centre(apical_dendrites()))

    i2p = sim.Projection(inputs, pyramidal_cells,
                        connector=sim.AllToAllConnector(location_selector=random_section(apical_dendrites())),
                        synapse_type=sim.StaticSynapse(weight=0.5, delay=0.5),
                        receptor_type="AMPA"
                        )

    sim.run(10.0)
    data = pyramidal_cells.get_data().segments[0]
