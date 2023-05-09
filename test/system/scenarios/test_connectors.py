
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import connection_plot
from pyNN.utility.script_tools import init_logging
from .fixtures import run_with_simulators
import pytest


#init_logging(None, debug=True)


# TODO: add some tests with projections between Assemblies and PopulationViews


@run_with_simulators("nest", "neuron", "brian2")
def test_all_to_all_static_no_self(sim):
    sim.setup()
    p = sim.Population(5, sim.IF_cond_exp())
    synapse_type = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2.0, theta=0.5), delay="0.2+0.3*d")
    prj = sim.Projection(p, p, sim.AllToAllConnector(allow_self_connections=False), synapse_type)
    weights = prj.get('weight', format='array', gather=False)
    print(weights)
    delays = prj.get('delay', format='list', gather=False)
    i, j, d = np.array(delays).T
    assert_allclose(d, 0.2 + 0.3 * abs(i - j), 1e-9)
    assert d.size == p.size * (p.size - 1)
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_all_to_all_tsodyksmarkram(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type = sim.TsodyksMarkramSynapse(weight=lambda d: d, delay=0.5, U=lambda d: 0.02 * d + 0.1)
    prj = sim.Projection(p1, p2, sim.AllToAllConnector(), synapse_type)
    i, j, w, d, u = np.array(prj.get(['weight', 'delay', 'U'], format='list', gather=False)).T
    assert_array_equal(w, abs(i - j))
    assert_array_equal(d, 0.5 * np.ones(p2.size * p1.size))
    assert_array_equal(u, 0.02 * abs(i - j) + 0.1)
    weights, delays, U = prj.get(['weight', 'delay', 'U'], format='array', gather=False)
    print(weights)
    print(delays)  # should all be 0.5
    print(U)
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_fixed_number_pre_no_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=False, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)

    print("Projection 1\n", connection_plot(prj1))
    weights1 = prj1.get('weight', format='array', gather=False)
    for column in weights1.T:
        assert (~np.isnan(column)).sum() == 3
        column[np.isnan(column)] = 0
        assert column.sum() == 1.5

    synapse_type2 = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2, theta=0.5), delay="0.2+0.3*d")
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("\nProjection 2\n", connection_plot(prj2))
    weights2 = prj2.get('weight', format='array', gather=False)
    delays2 = prj2.get('delay', format='list', gather=False)
    print(weights2)
    print(delays2)
    for i, j, d in delays2:
        assert d == pytest.approx(0.2 + 0.3 * abs(i - j))  # places=9
    for column in weights2.T:
        assert (~np.isnan(column)).sum() == 3
        column[np.isnan(column)] = 0
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_fixed_number_pre_with_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=True, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)
    print("Projection #1\n", connection_plot(prj1))
    delays = prj1.get('delay', format='list', gather=False)
    assert len(delays) == connector1.n * p2.size
    weights = prj1.get('weight', format='array', gather=False)
    for column in weights.T:
        column[np.isnan(column)] = 0
        assert column.sum() == 1.5


@run_with_simulators("nest", "neuron", "brian2")
def test_fixed_number_pre_with_replacement_heterogeneous_parameters(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    connector1 = sim.FixedNumberPreConnector(n=3, with_replacement=True, rng=NumpyRNG())
    synapse_type2 = sim.TsodyksMarkramSynapse(weight=lambda d: d, delay=0.5, U=lambda d: 0.02 * d + 0.1)
    #synapse_type2 = sim.TsodyksMarkramSynapse(weight=0.001, delay=0.5, U=lambda d: 2*d+0.1)
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("Projection 2")
    x = prj2.get(['weight', 'delay', 'U'], format='list', gather=False)
    from pprint import pprint
    pprint(x)
    i, j, w, d, u = np.array(x).T
    assert_array_equal(w, abs(i - j))
    assert_array_equal(d, 0.5 * np.ones(p2.size * connector1.n))
    assert_array_equal(u, 0.02 * abs(i - j) + 0.1)
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_fixed_number_post_no_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPostConnector(n=3, with_replacement=False, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)

    print("Projection 1\n", connection_plot(prj1))
    weights1 = prj1.get('weight', format='array', gather=False)
    for row in weights1:
        assert (~np.isnan(row)).sum() == 3
        row[np.isnan(row)] = 0
        assert row.sum() == 1.5

    synapse_type2 = sim.StaticSynapse(weight=RandomDistribution('gamma', k=2, theta=0.5), delay="0.2+0.3*d")
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("\nProjection 2\n", connection_plot(prj2))
    weights2 = prj2.get('weight', format='array', gather=False)
    delays2 = prj2.get('delay', format='list', gather=False)
    print(weights2)
    print(delays2)
    for i, j, d in delays2:
        assert d == pytest.approx(0.2 + 0.3 * abs(i - j))  # places=9
    for row in weights2:
        assert (~np.isnan(row)).sum() == 3
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_fixed_number_post_with_replacement(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    synapse_type1 = sim.StaticSynapse(weight=0.5, delay=0.5)
    connector1 = sim.FixedNumberPostConnector(n=9, with_replacement=True, rng=NumpyRNG())
    prj1 = sim.Projection(p1, p2, connector1, synapse_type1)
    print("Projection #1\n", connection_plot(prj1))
    delays = prj1.get('delay', format='list', gather=False)
    assert len(delays) == connector1.n * p1.size
    weights = prj1.get('weight', format='array', gather=False)
    for row in weights:
        row[np.isnan(row)] = 0
        assert row.sum() == 4.5

    weights2 = prj1.get('weight', format='array', gather=False, multiple_synapses='min')
    for row in weights2:
        n_nan = np.isnan(row).sum()
        row[np.isnan(row)] = 0
        assert row.sum() == (row.size - n_nan)*0.5


@run_with_simulators("nest", "neuron", "brian2")
def test_fixed_number_post_with_replacement_heterogeneous_parameters(sim):
    sim.setup()
    p1 = sim.Population(5, sim.IF_cond_exp())
    p2 = sim.Population(7, sim.IF_cond_exp())
    connector1 = sim.FixedNumberPostConnector(n=3, with_replacement=True, rng=NumpyRNG())
    synapse_type2 = sim.TsodyksMarkramSynapse(weight=lambda d: d, delay=0.5, U=lambda d: 0.02 * d + 0.1)
    #synapse_type2 = sim.TsodyksMarkramSynapse(weight=0.001, delay=0.5, U=lambda d: 2*d+0.1)
    prj2 = sim.Projection(p1, p2, connector1, synapse_type2)
    print("Projection 2")
    x = prj2.get(['weight', 'delay', 'U'], format='list', gather=False)
    from pprint import pprint
    pprint(x)
    i, j, w, d, u = np.array(x).T
    assert_array_equal(w, abs(i - j))
    assert_array_equal(d, 0.5 * np.ones(p1.size * connector1.n))
    assert_array_equal(u, 0.02 * abs(i - j) + 0.1)
    sim.end()


@run_with_simulators("nest", "neuron", "brian2")
def test_issue309(sim):
    # check that FixedProbability(1) gives the same results as AllToAll
    sim.setup()
    p = sim.Population(5, sim.IF_cond_exp())
    synapse_type = sim.StaticSynapse(weight=0.1, delay="0.2+0.3*d")
    prj_a2a = sim.Projection(p, p, sim.AllToAllConnector(allow_self_connections=False), synapse_type)
    prj_fp1 = sim.Projection(p, p, sim.FixedProbabilityConnector(p_connect=1, allow_self_connections=False), synapse_type)
    assert sorted(prj_a2a.get('weight', format='list', gather=False)) == \
                 sorted(prj_fp1.get('weight', format='list', gather=False))
    assert sorted(prj_a2a.get('delay', format='list', gather=False)) == \
                 sorted(prj_fp1.get('delay', format='list', gather=False))
    assert prj_fp1.size() == 20  # 20 rather than 25 because self-connections are excluded
    sim.end()


@run_with_simulators("nest", "neuron")
def test_issue622(sim):
    sim.setup()
    pop = sim.Population(10, sim.IF_cond_exp, {}, label="pop")

    view1 = sim.PopulationView(pop, [2, 3, 4])
    view2 = sim.PopulationView(pop, [2, 3, 4])

    proj1 = sim.Projection(view1, view2,
                           sim.AllToAllConnector(allow_self_connections=False),
                           sim.StaticSynapse(weight=0.015, delay=1.0), receptor_type='excitatory')
    proj2 = sim.Projection(view1, view1,
                           sim.AllToAllConnector(allow_self_connections=False),
                           sim.StaticSynapse(weight=0.015, delay=1.0), receptor_type='excitatory')

    w1 = proj1.get("weight", "list")
    w2 = proj2.get("weight", "list")

    assert set(w1) == set(w2)
    assert set(w1) == set([(0.0, 1.0, 0.015), (0.0, 2.0, 0.015), (1.0, 0.0, 0.015),
                           (1.0, 2.0, 0.015), (2.0, 0.0, 0.015), (2.0, 1.0, 0.015)])

    # partially overlapping views
    print("Now with partial overlap")
    view3 = sim.PopulationView(pop, [3, 4, 5, 6])

    proj3 = sim.Projection(view1, view3,
                           sim.AllToAllConnector(allow_self_connections=False),
                           sim.StaticSynapse(weight=0.015, delay=1.0), receptor_type='excitatory')

    w3 = proj3.get("weight", "list")
    assert set(w3) == \
                 set([
                     (0.0, 0.0, 0.015), (0.0, 1.0, 0.015), (0.0, 2.0, 0.015), (0.0, 3.0, 0.015),
                                        (1.0, 1.0, 0.015), (1.0, 2.0, 0.015), (1.0, 3.0, 0.015),
                     (2.0, 0.0, 0.015),                    (2.0, 2.0, 0.015), (2.0, 3.0, 0.015)
                 ])

    view4 = sim.PopulationView(pop, [0, 1])
    assmbl = view3 + view4
    proj4 = sim.Projection(view1, assmbl,
                           sim.FixedProbabilityConnector(p_connect=0.99999, allow_self_connections=False),
                           sim.StaticSynapse(weight=0.015, delay=1.0), receptor_type='excitatory')
    w4 = proj4.get("weight", "list")
    assert set(w4) == \
                 set([
                     (0, 0, 0.015), (0, 1, 0.015), (0, 2, 0.015), (0, 3, 0.015), (0, 4, 0.015), (0, 5, 0.015),
                                    (1, 1, 0.015), (1, 2, 0.015), (1, 3, 0.015), (1, 4, 0.015), (1, 5, 0.015),
                     (2, 0, 0.015),                (2, 2, 0.015), (2, 3, 0.015), (2, 4, 0.015), (2, 5, 0.015),
                 ])


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_all_to_all_static_no_self(sim)
    test_all_to_all_tsodyksmarkram(sim)
    test_fixed_number_pre_no_replacement(sim)
    test_fixed_number_pre_with_replacement(sim)
    test_fixed_number_pre_with_replacement_heterogeneous_parameters(sim)
    test_issue309(sim)
