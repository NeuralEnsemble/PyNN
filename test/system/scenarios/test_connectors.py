
import numpy
from nose.tools import assert_equal
from pyNN.random import NumpyRNG
from pyNN.utility import assert_arrays_equal, connection_plot, init_logging
from registry import register

#init_logging(None, debug=True)

@register()
def fixed_number_pre_no_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=False, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)
    print connection_plot(prj1)
    weights = prj1.get('weight', format='array', gather=False)
    for column in weights.T:
        assert_equal((~numpy.isnan(column)).sum(), 3)
        column[numpy.isnan(column)] = 0
        assert_equal(column.sum(), 1.5)
    sim.end()


@register()
def fixed_number_pre_with_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=True, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)
    print connection_plot(prj1)
    delays = prj1.get('delay', format='list', gather=False)
    assert_equal(len(delays), connector1.n * p2.size)
    weights = prj1.get('weight', format='array', gather=False)
    for column in weights.T:
        column[numpy.isnan(column)] = 0
        assert_equal(column.sum(), 1.5)
    sim.end()


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    fixed_number_pre_no_replacement(sim)
    fixed_number_pre_with_replacement(sim)
