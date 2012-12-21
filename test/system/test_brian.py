from nose.plugins.skip import SkipTest
from scenarios import scenarios

try:
    import pyNN.brian
    have_brian = True
except ImportError:
    have_brian = False

def test_scenarios():
    for scenario in scenarios:
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
    p1 = pynnn.Population(9, pynnn.IF_curr_alpha, structure=pynnn.space.Grid2D())
    p2 = pynnn.Population(9, pynnn.IF_curr_alpha, structure=pynnn.space.Grid2D())
    p1.record(to_file=False)
    p2.record(to_file=False)
    prj1_2 = pynnn.Projection(p1, p2, pynnn.OneToOneConnector(weights=10.0), target='excitatory')
    # we note that the connectivity is as expected: a uniform diagonal
    prj1_2.getWeights(format='array')
    src = pynnn.DCSource(amplitude=70)
    src.inject_into(p1[:])
    pynnn.run(50)
    n_spikes_p1 = p1.get_spike_counts()
    # We see that both populations have fired uniformly as expected:
    n_spikes_p2 = p2.get_spike_counts()
    for n in n_spikes_p1.values():
        assert n == n_spikes_p1[p1[1]]
    for n in n_spikes_p2.values():
        assert n == n_spikes_p2[p2[1]]
    # With this new setup, only the second p2 unit should fire:
    prj1_2.setWeights([0, 20, 0, 0, 0, 0, 0, 0, 0])
    # This looks good:
    prj1_2.getWeights(format='array')
    pynnn.run(50)
    n_spikes_p1 = p1.get_spike_counts()
    for n in n_spikes_p1.values():
        assert n == n_spikes_p1[p1[1]]
    # p2[1] should be ahead in spikes count, and others should not have
    # fired more. It is not what I observe:
    n_spikes_p2 = p2.get_spike_counts()
    assert n_spikes_p2[p2[1]] > n_spikes_p2[p2[0]]
