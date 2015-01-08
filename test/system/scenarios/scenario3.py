# encoding: utf-8

from nose.tools import assert_equal
from nose.plugins.skip import SkipTest
from pyNN.utility import init_logging
from pyNN.random import RandomDistribution
from .registry import register


@register(exclude=["moose", "nemo", "brian"])
def scenario3(sim):
    """
    Simple feed-forward network network with additive STDP. The second half of
    the presynaptic neurons fires faster than the second half, so their
    connections should be potentiated more.
    """

    init_logging(logfile=None, debug=True)
    second = 1000.0
    duration = 10
    tau_m = 20 # ms
    cm = 1.0 # nF
    v_reset = -60
    cell_parameters = dict(
        tau_m = tau_m,
        cm = cm,
        v_rest = -70,
        e_rev_E = 0,
        e_rev_I = -70,
        v_thresh = -54,
        v_reset = v_reset,
        tau_syn_E = 5,
        tau_syn_I = 5,
    )
    g_leak = cm/tau_m # µS

    w_min = 0.0*g_leak
    w_max = 0.05*g_leak

    r1 = 5.0
    r2 = 40.0

    sim.setup()
    pre = sim.Population(100, sim.SpikeSourcePoisson())
    post = sim.Population(10, sim.IF_cond_exp())

    pre.set(duration=duration*second)
    pre.set(start=0.0)
    pre[:50].set(rate=r1)
    pre[50:].set(rate=r2)
    assert_equal(pre[49].rate, r1)
    assert_equal(pre[50].rate, r2)
    post.set(**cell_parameters)
    post.initialize(v=RandomDistribution('normal', mu=v_reset, sigma=5.0))

    stdp = sim.STDPMechanism(
                sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                  A_plus=0.01, A_minus=0.01),
                sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max),
                #dendritic_delay_fraction=0.5))
                dendritic_delay_fraction=1)

    connections = sim.Projection(pre, post, sim.AllToAllConnector(),
                                 synapse_type=stdp,
                                 receptor_type='excitatory')

    initial_weight_distr = RandomDistribution('uniform', low=w_min, high=w_max)
    connections.randomizeWeights(initial_weight_distr)
    initial_weights = connections.get('weight', format='array', gather=False)
    assert initial_weights.min() >= w_min
    assert initial_weights.max() < w_max
    assert initial_weights[0,0] != initial_weights[1,0]

    pre.record('spikes')
    post.record('spikes')
    post[0:1].record('v')

    sim.run(duration*second)

    actual_rate = pre.mean_spike_count()/duration
    expected_rate = (r1+r2)/2
    errmsg = "actual rate: %g  expected rate: %g" % (actual_rate, expected_rate)
    assert abs(actual_rate - expected_rate) < 1, errmsg
    #assert abs(pre[:50].mean_spike_count()/duration - r1) < 1
    #assert abs(pre[50:].mean_spike_count()/duration- r2) < 1
    final_weights = connections.get('weight', format='array', gather=False)
    assert initial_weights[0,0] != final_weights[0,0]

    try:
        import scipy.stats
    except ImportError:
        raise SkipTest
    t,p = scipy.stats.ttest_ind(initial_weights[:50,:].flat, initial_weights[50:,:].flat)
    assert p > 0.05, p
    t,p = scipy.stats.ttest_ind(final_weights[:50,:].flat, final_weights[50:,:].flat)
    assert p < 0.01, p
    assert final_weights[:50,:].mean() < final_weights[50:,:].mean()
    sim.end()
    return initial_weights, final_weights, pre, post, connections
    

if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    scenario3(sim)
