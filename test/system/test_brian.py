from nose.plugins.skip import SkipTest
from nose.tools import assert_equal
import numpy
from numpy.testing import assert_array_equal
from scenarios.registry import registry

try:
    import pyNN.brian
    have_brian = True
except ImportError:
    have_brian = False

def test_scenarios():
    for scenario in registry:
        if "brian" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_brian:
                yield scenario, pyNN.brian
            else:
                raise SkipTest


def test_ticket235():
    if not have_brian:
        raise SkipTest
    pynnn = pyNN.brian
    pynnn.setup()
    p1 = pynnn.Population(9, pynnn.IF_curr_alpha(), structure=pynnn.space.Grid2D())
    p2 = pynnn.Population(9, pynnn.IF_curr_alpha(), structure=pynnn.space.Grid2D())
    p1.record('spikes', to_file=False)
    p2.record('spikes', to_file=False)
    prj1_2 = pynnn.Projection(p1, p2, pynnn.OneToOneConnector(), pynnn.StaticSynapse(weight=10.0), receptor_type='excitatory')
    # we note that the connectivity is as expected: a uniform diagonal
    prj1_2.get('weight', format='array')
    src = pynnn.DCSource(amplitude=70)
    src.inject_into(p1[:])
    pynnn.run(50)
    n_spikes_p1 = p1.get_spike_counts()
    # We see that both populations have fired uniformly as expected:
    n_spikes_p2 = p2.get_spike_counts()
    for n in n_spikes_p1.values():
        assert_equal(n, n_spikes_p1[p1[1]])
    for n in n_spikes_p2.values():
        assert_equal(n, n_spikes_p2[p2[1]])
    # With this new setup, only the second p2 unit should fire:
    #prj1_2.set(weight=[0, 20, 0, 0, 0, 0, 0, 0, 0])
    new_weights = numpy.where(numpy.eye(9), 0, numpy.nan)
    new_weights[1, 1] = 20.0
    prj1_2.set(weight=new_weights)
    # This looks good:
    prj1_2.get('weight', format='array')
    pynnn.run(50)
    n_spikes_p1 = p1.get_spike_counts()
    for n in n_spikes_p1.values():
        assert_equal(n, n_spikes_p1[p1[1]])
    # p2[1] should be ahead in spikes count, and others should not have
    # fired more. It is not what I observe:
    n_spikes_p2 = p2.get_spike_counts()
    assert n_spikes_p2[p2[1]] > n_spikes_p2[p2[0]]


def test_tsodyks_markram_synapse():
    if not have_brian:
        raise SkipTest
    sim = pyNN.brian
    sim.setup()
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=numpy.arange(10, 100, 10)))
    neurons = sim.Population(5, sim.IF_cond_exp(e_rev_I=-75, tau_syn_I=numpy.arange(0.2, 0.7, 0.1)))
    synapse_type = sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0,
                                             tau_facil=1000.0, weight=0.01,
                                             delay=0.5)
    connector = sim.AllToAllConnector()
    prj = sim.Projection(spike_source, neurons, connector,
                         receptor_type='inhibitory',
                         synapse_type=synapse_type)
    neurons.record('gsyn_inh')
    sim.run(100.0)
    tau_psc = prj._brian_synapses[0][0].tau_syn.data * 1e3  # s --> ms
    assert_array_equal(tau_psc, numpy.arange(0.2, 0.7, 0.1))