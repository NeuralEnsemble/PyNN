
import numpy
from nose.tools import assert_equal, assert_almost_equal
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import assert_arrays_equal, assert_arrays_almost_equal, connection_plot, init_logging
from .registry import register

#init_logging(None, debug=True)


# TODO: add some tests with projections between Assemblies and PopulationViews


@register()
def all_to_all_static_no_self(sim):
    sim.setup()
    p = sim.Population(5, sim.IF_cond_exp())
    synapse_type = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2.0, theta=0.5), delay="0.2+0.3*d")
    prj = sim.Projection(p, p, sim.AllToAllConnector(allow_self_connections=False), synapse_type)
    weights = prj.get('weight', format='array', gather=False)
    print(weights)
    delays = prj.get('delay', format='list', gather=False)
    i, j, d = numpy.array(delays).T
    assert_arrays_almost_equal(d, 0.2 + 0.3*abs(i - j), 1e-9)
    assert_equal(d.size, p.size*(p.size - 1))
    sim.end()


@register()
def all_to_all_tsodyksmarkram(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type = sim.TsodyksMarkramSynapse(weight=lambda d: d, delay=0.5, U=lambda d: 2*d+0.1)
    prj = sim.Projection(p1, p2, sim.AllToAllConnector(), synapse_type)
    i, j, w, d, u = numpy.array(prj.get(['weight', 'delay', 'U'], format='list', gather=False)).T
    assert_arrays_equal(w, abs(i - j))
    assert_arrays_equal(d, 0.5*numpy.ones(p2.size * p1.size))
    assert_arrays_equal(u, 2*abs(i-j) + 0.1)
    weights, delays, U = prj.get(['weight', 'delay', 'U'], format='array', gather=False)
    print(weights)
    print(delays)  # should all be 0.5
    print(U)
    sim.end()


@register()
def fixed_number_pre_no_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=False, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)

    print("Projection 1\n", connection_plot(prj1))
    weights1 = prj1.get('weight', format='array', gather=False)
    for column in weights1.T:
        assert_equal((~numpy.isnan(column)).sum(), 3)
        column[numpy.isnan(column)] = 0
        assert_equal(column.sum(), 1.5)

    synapse_type2 = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2, theta=0.5), delay="0.2+0.3*d")
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("\nProjection 2\n", connection_plot(prj2))
    weights2 = prj2.get('weight', format='array', gather=False)
    delays2 = prj2.get('delay', format='list', gather=False)
    print(weights2)
    print(delays2)
    for i, j, d in delays2:
        assert_almost_equal(d, 0.2 + 0.3*abs(i - j), 9)
    for column in weights2.T:
        assert_equal((~numpy.isnan(column)).sum(), 3)
        column[numpy.isnan(column)] = 0
    sim.end()


@register()
def fixed_number_pre_with_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=True, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)
    print("Projection #1\n", connection_plot(prj1))
    delays = prj1.get('delay', format='list', gather=False)
    assert_equal(len(delays), connector1.n * p2.size)
    weights = prj1.get('weight', format='array', gather=False)
    for column in weights.T:
        column[numpy.isnan(column)] = 0
        assert_equal(column.sum(), 1.5)


@register()
def fixed_number_pre_with_replacement_heterogeneous_parameters(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=True, rng=NumpyRNG())
    synapse_type2 = sim.TsodyksMarkramSynapse(weight=lambda d: d, delay=0.5, U=lambda d: 2*d+0.1)
    #synapse_type2 = sim.TsodyksMarkramSynapse(weight=0.001, delay=0.5, U=lambda d: 2*d+0.1)
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("Projection 2")
    x = prj2.get(['weight', 'delay', 'U'], format='list', gather=False)
    from pprint import pprint
    pprint(x)
    i, j, w, d, u = numpy.array(x).T
    assert_arrays_equal(w, abs(i - j))
    assert_arrays_equal(d, 0.5*numpy.ones(p2.size * connector1.n))
    assert_arrays_equal(u, 2*abs(i-j) + 0.1)
    sim.end()


@register()
def fixed_number_post_no_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPostConnector(n=3, with_replacement=False, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)

    print("Projection 1\n", connection_plot(prj1))
    weights1 = prj1.get('weight', format='array', gather=False)
    for row in weights1:
        assert_equal((~numpy.isnan(row)).sum(), 3)
        row[numpy.isnan(row)] = 0
        assert_equal(row.sum(), 1.5)

    synapse_type2 = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2, theta=0.5), delay="0.2+0.3*d")
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("\nProjection 2\n", connection_plot(prj2))
    weights2 = prj2.get('weight', format='array', gather=False)
    delays2 = prj2.get('delay', format='list', gather=False)
    print(weights2)
    print(delays2)
    for i, j, d in delays2:
        assert_almost_equal(d, 0.2 + 0.3*abs(i - j), 9)
    for row in weights2:
        assert_equal((~numpy.isnan(row)).sum(), 3)
    sim.end()

@register()
def fixed_number_post_with_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPostConnector(n=9, with_replacement=True, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)
    print("Projection #1\n", connection_plot(prj1))
    delays = prj1.get('delay', format='list', gather=False)
    assert_equal(len(delays), connector1.n * p1.size)
    weights = prj1.get('weight', format='array', gather=False)
    for row in weights:
        row[numpy.isnan(row)] = 0
        assert_equal(row.sum(), 4.5)

@register()
def fixed_number_post_with_replacement_heterogeneous_parameters(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    connector1 = sim.FixedNumberPostConnector(n=3, with_replacement=True, rng=NumpyRNG())
    synapse_type2 = sim.TsodyksMarkramSynapse(weight=lambda d: d, delay=0.5, U=lambda d: 2*d+0.1)
    #synapse_type2 = sim.TsodyksMarkramSynapse(weight=0.001, delay=0.5, U=lambda d: 2*d+0.1)
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("Projection 2")
    x = prj2.get(['weight', 'delay', 'U'], format='list', gather=False)
    from pprint import pprint
    pprint(x)
    i, j, w, d, u = numpy.array(x).T
    assert_arrays_equal(w, abs(i - j))
    assert_arrays_equal(d, 0.5*numpy.ones(p1.size * connector1.n))
    assert_arrays_equal(u, 2*abs(i-j) + 0.1)
    sim.end()


@register()
def issue309(sim):
    # check that FixedProbability(1) gives the same results as AllToAll
    sim.setup()
    p = sim.Population(5, sim.IF_cond_exp())
    synapse_type = sim.StaticSynapse(weight=0.1, delay="0.2+0.3*d")
    prj_a2a = sim.Projection(p, p, sim.AllToAllConnector(allow_self_connections=False), synapse_type)
    prj_fp1 = sim.Projection(p, p, sim.FixedProbabilityConnector(p_connect=1, allow_self_connections=False), synapse_type)
    assert_equal(prj_a2a.get('weight', format='list', gather=False),
                 prj_fp1.get('weight', format='list', gather=False))
    assert_equal(prj_a2a.get('delay', format='list', gather=False),
                 prj_fp1.get('delay', format='list', gather=False))
    assert_equal(prj_fp1.size(), 20)  # 20 rather than 25 because self-connections are excluded
    sim.end()


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    all_to_all_static_no_self(sim)
    all_to_all_tsodyksmarkram(sim)
    fixed_number_pre_no_replacement(sim)
    fixed_number_pre_with_replacement(sim)
    fixed_number_pre_with_replacement_heterogeneous_parameters(sim)
    issue309(sim)