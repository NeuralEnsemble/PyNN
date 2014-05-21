from nose.plugins.skip import SkipTest
from nose.tools import assert_equal
import numpy
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
