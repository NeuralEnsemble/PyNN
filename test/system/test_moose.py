from nose.plugins.skip import SkipTest
from scenarios.registry import registry
from nose.tools import assert_equal, assert_almost_equal
from pyNN.random import RandomDistribution
from pyNN.utility import init_logging

try:
    import pyNN.moose
    have_moose = True
except ImportError:
    have_moose = False


def test_scenarios():
    for scenario in registry:
        if "moose" not in scenario.exclude:
            scenario.description = scenario.__name__
            if have_moose:
                yield scenario, pyNN.moose
            else:
                raise SkipTest

def test_recording():
    if not have_moose:
        raise SkipTest

    sim = pyNN.moose
    sim.setup()

    p = sim.Population(2, sim.HH_cond_exp, {'i_offset': 0.1})
    p.initialize(v=-65.0)
    p.record_v()

    sim.run(100.0)

    id, t, v = p.get_v().T
    assert v.max() > 0 # at least one spike
    sim.end()

    return id, t, v


def test_synaptic_connections():
    if not have_moose:
        raise SkipTest

    import numpy
    sim = pyNN.moose
    sim.setup()

    p1 = sim.Population(1, sim.SpikeSourceArray, {'spike_times': numpy.arange(3.0, 103, 10.0)})
    #p1 = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 100.0})
    #p1 = sim.Population(1, sim.HH_cond_exp, {'i_offset': 1.0})
    #p2 = sim.Population(1, sim.HH_cond_exp)
    p2 = sim.Population(1, sim.IF_cond_exp)

    prj = sim.Projection(p1, p2, sim.AllToAllConnector(weights=1e-4))

    p2.record_v()

    sim.run(100.0)

    id, t, v2 = p2.get_v().T
    sim.end()

    return id, t, v2