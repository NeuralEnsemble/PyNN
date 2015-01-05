"""
Tests of the Connector classes, using the pyNN.mock backend.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from pyNN import connectors, random, errors, space, recording
import numpy
import os
import sys
from numpy.testing import assert_array_equal, assert_array_almost_equal
from .mocks import MockRNG, MockRNG2
import pyNN.mock as sim

from .backends.registry import register_class, register

orig_mpi_get_config = random.get_mpi_config


def setUp():
    random.get_mpi_config = lambda: (0, 2)


def tearDown():
    random.get_mpi_config = orig_mpi_get_config
        
@register_class()
class TestOneToOneConnector(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        sim.setup(num_processes=2, rank=0, **extra)
        self.p1 = sim.Population(5, sim.IF_cond_exp())
        self.p2 = sim.Population(5, sim.HH_cond_exp())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        
    def tearDown(self, sim=sim):
        sim.end()

    @register()
    def test_connect_with_scalar_weights_and_delays(self, sim=sim):
        C = connectors.OneToOneConnector(safe=False)
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 1, 5.0, 0.5),
                          (3, 3, 5.0, 0.5)])

    @register()
    def test_connect_with_random_weights(self, sim=sim):
        rd = random.RandomDistribution('uniform', (0, 1), rng=MockRNG(delta=1.0))
        syn = sim.StaticSynapse(weight=rd, delay=0.5)
        C = connectors.OneToOneConnector(safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 1, 1.0, 0.5),
                          (3, 3, 3.0, 0.5)])


@register_class()
class TestAllToAllConnector(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        sim.setup(num_processes=2, rank=1, min_delay=0.123, **extra)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p1._mask_local, numpy.array([0,1,0,1], dtype=bool))
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    @register()
    def test_connect_with_scalar_weights_and_delays(self, sim=sim):
        C = connectors.AllToAllConnector(safe=False)
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 5.0, 0.5),
                          (1, 1, 5.0, 0.5),
                          (2, 1, 5.0, 0.5),
                          (3, 1, 5.0, 0.5),
                          (0, 3, 5.0, 0.5),
                          (1, 3, 5.0, 0.5),
                          (2, 3, 5.0, 0.5),
                          (3, 3, 5.0, 0.5)])
        nan = numpy.nan
        assert_array_equal(prj.get('weight', format='array', gather=False),
                           numpy.array([[nan, 5.0, nan, 5.0, nan],
                                        [nan, 5.0, nan, 5.0, nan],
                                        [nan, 5.0, nan, 5.0, nan],
                                        [nan, 5.0, nan, 5.0, nan]]))

    @register()
    def test_connect_with_array_weights(self, sim=sim):
        C = connectors.AllToAllConnector(safe=False)
        syn = sim.StaticSynapse(weight=numpy.arange(0.0, 2.0, 0.1).reshape(4, 5), delay=0.5)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        assert_array_almost_equal(
            numpy.array(prj.get(["weight", "delay"], format='list', gather=False)),  # use gather False because we are faking the MPI
            numpy.array([(0, 1, 0.1, 0.5),
                         (1, 1, 0.6, 0.5),
                         (2, 1, 1.1, 0.5),
                         (3, 1, 1.6, 0.5),
                         (0, 3, 0.3, 0.5),
                         (1, 3, 0.8, 0.5),
                         (2, 3, 1.3, 0.5),
                         (3, 3, 1.8, 0.5)]))
        nan = numpy.nan
        assert_array_almost_equal(prj.get('weight', format='array', gather=False),
                                  numpy.array([[nan, 0.1, nan, 0.3, nan],
                                               [nan, 0.6, nan, 0.8, nan],
                                               [nan, 1.1, nan, 1.3, nan],
                                               [nan, 1.6, nan, 1.8, nan]]),
                                  9)

    @register()
    def test_connect_with_random_weights_parallel_safe(self, sim=sim):
        rd = random.RandomDistribution('uniform', (0, 1), rng=MockRNG(delta=1.0, parallel_safe=True))
        syn = sim.StaticSynapse(weight=rd, delay=0.5)
        C = connectors.AllToAllConnector(safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        # note that the outer loop is over the post-synaptic cells, the inner loop over the pre-synaptic
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 4.0, 0.5),
                          (1, 1, 5.0, 0.5),
                          (2, 1, 6.0, 0.5),
                          (3, 1, 7.0, 0.5),
                          (0, 3, 12.0, 0.5),
                          (1, 3, 13.0, 0.5),
                          (2, 3, 14.0, 0.5),
                          (3, 3, 15.0, 0.5)])
        nan = numpy.nan
        assert_array_almost_equal(prj.get('weight', format='array', gather=False),
                                  numpy.array([[nan, 4.0, nan, 12.0, nan],
                                               [nan, 5.0, nan, 13.0, nan],
                                               [nan, 6.0, nan, 14.0, nan],
                                               [nan, 7.0, nan, 15.0, nan]]),
                                  9)

    @register()
    def test_connect_with_distance_dependent_weights(self, sim=sim):
        d_expr = "d+100"
        syn = sim.StaticSynapse(weight=d_expr, delay=0.5)
        C = connectors.AllToAllConnector(safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 101.0, 0.5),
                          (1, 1, 100.0, 0.5),
                          (2, 1, 101.0, 0.5),
                          (3, 1, 102.0, 0.5),
                          (0, 3, 103.0, 0.5),
                          (1, 3, 102.0, 0.5),
                          (2, 3, 101.0, 0.5),
                          (3, 3, 100.0, 0.5)])

    @register()
    def test_connect_with_distance_dependent_weights_and_delays(self, sim=sim):
        syn = sim.StaticSynapse(weight="d+100", delay="0.2+2*d")
        C = connectors.AllToAllConnector(safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 101.0, 2.2),
                          (1, 1, 100.0, 0.2),
                          (2, 1, 101.0, 2.2),
                          (3, 1, 102.0, 4.2),
                          (0, 3, 103.0, 6.2),
                          (1, 3, 102.0, 4.2),
                          (2, 3, 101.0, 2.2),
                          (3, 3, 100.0, 0.2)])

    @register()
    def test_connect_with_delays_None(self, sim=sim):
        syn = sim.StaticSynapse(weight=0.1, delay=None)
        C = connectors.AllToAllConnector()
        assert C.safe
        assert C.allow_self_connections
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False)[0][3], prj._simulator.state.min_delay)

    @register()
    @unittest.skip('skipping this tests until I figure out how I want to refactor checks')
    def test_connect_with_delays_too_small(self, sim=sim):
        C = connectors.AllToAllConnector()
        syn = sim.StaticSynapse(weight=0.1, delay=0.0)
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C, syn)

    @register()
    @unittest.skip('skipping this tests until I figure out how I want to refactor checks')
    def test_connect_with_list_delays_too_small(self, sim=sim):
        delays = numpy.ones((self.p1.size, self.p2.size), float)
        delays[2, 3] = sim.Projection._simulator.state.min_delay - 0.01
        syn = sim.StaticSynapse(weight=0.1, delay=delays)
        C = connectors.AllToAllConnector()
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C, syn)


@register_class()
class TestFixedProbabilityConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    @register()
    def test_connect_with_default_args(self, sim=sim):
        C = connectors.FixedProbabilityConnector(p_connect=0.85,
                                                 rng=MockRNG(delta=0.1, parallel_safe=True))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)

        # 20 possible connections. Due to the mock RNG, only the
        # first 9 are created (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2)
        # of these, (0,1), (1,1), (2,1), (3,1) are created on this node
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123)])

    @register()
    def test_connect_with_default_args_again(self, sim=sim):
        C = connectors.FixedProbabilityConnector(p_connect=0.5,
                                                 rng=MockRNG2(1 - numpy.array([1, 0, 0, 1,
                                                                               0, 0, 0, 1,
                                                                               1, 1, 0, 0,
                                                                               1, 0, 1, 0,
                                                                               1, 1, 0, 1]),
                                                              parallel_safe=True))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)

        # 20 possible connections. Due to the mock RNG, only the following
        # are created (0,0), (3,0), (3,1), (0,2), (1,2), (0,3), (2,3), (0,4), (1,4), (3,4)
        # of these, (3,1), (0,3), (2,3) are created on this node
        # (note that the outer loop is over post-synaptic cells (columns), the inner loop over pre-synaptic (rows))
        nan = numpy.nan
        assert_array_almost_equal(prj.get('delay', format='array', gather=False),
                                  numpy.array([[nan, nan,   nan, 0.123, nan],
                                               [nan, nan,   nan, nan,   nan],
                                               [nan, nan,   nan, 0.123, nan],
                                               [nan, 0.123, nan, nan,   nan]]),
                                  9)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(3, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123)])

    @register()
    def test_connect_with_probability_1(self, sim=sim):
        # see https://github.com/NeuralEnsemble/PyNN/issues/309
        C = connectors.FixedProbabilityConnector(p_connect=1,
                                                 rng=MockRNG(delta=0.01, parallel_safe=True))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)

        # 20 connections, only some of which are created on this node
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123)
                         ])

    @register()
    def test_connect_with_weight_function(self, sim=sim):
        C = connectors.FixedProbabilityConnector(p_connect=0.85,
                                                 rng=MockRNG(delta=0.1))
        syn = sim.StaticSynapse(weight=lambda d: 0.1*d)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.1, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.1, 0.123),
                          (3, 1, 0.2, 0.123)])

    @register()
    def test_connect_with_random_delays_parallel_safe(self, sim=sim):
        rd = random.RandomDistribution('uniform', low=0.1, high=1.1, rng=MockRNG(start=1.0, delta=0.2, parallel_safe=True))
        syn = sim.StaticSynapse(delay=rd)
        C = connectors.FixedProbabilityConnector(p_connect=0.5,
                                                 rng=MockRNG2(1 - numpy.array([1, 0, 0, 1,
                                                                               0, 0, 0, 1,
                                                                               1, 1, 0, 0,
                                                                               1, 0, 1, 0,
                                                                               1, 1, 0, 1]),
                                                              parallel_safe=True))
        prj = sim.Projection(self.p1, self.p2, C, syn)
        nan = numpy.nan
        assert_array_almost_equal(prj.get('delay', format='array', gather=False),
                                  numpy.array([[nan, nan, nan, 2.0, nan],
                                               [nan, nan, nan, nan, nan],
                                               [nan, nan, nan, 2.2, nan],
                                               [nan, 1.4, nan, nan, nan]]),
                                  9)

    #def test_connect_with_random_delays_parallel_unsafe(self, sim=sim):
    #    rd = random.RandomDistribution('uniform', [0.1, 1.1], rng=MockRNG(start=1.0, delta=0.2, parallel_safe=False))
    #    syn = sim.StaticSynapse(delay=rd)
    #    C = connectors.FixedProbabilityConnector(p_connect=0.5,
    #                                             rng=MockRNG2(1 - numpy.array([1, 0, 0, 1,
    #                                                                           0, 0, 0, 1,
    #                                                                           1, 1, 0, 0,
    #                                                                           1, 0, 1, 0,
    #                                                                           1, 1, 0, 1]),
    #                                                          parallel_safe=False))
    #    prj = sim.Projection(self.p1, self.p2, C, syn)
    #    nan = numpy.nan
    #    assert_array_almost_equal(prj.get('delay', format='array', gather=False),
    #                              numpy.array([[nan, nan, nan, 1.2, nan],
    #                                           [nan, nan, nan, nan, nan],
    #                                           [nan, nan, nan, 1.4, nan],
    #                                           [nan, 1.0, nan, nan, nan]]),
    #                              9)


@register_class()
class TestDistanceDependentProbabilityConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    @register()
    def test_connect_with_default_args(self, sim=sim):
        C = connectors.DistanceDependentProbabilityConnector(d_expression="d<1.5",
                                                             rng=MockRNG(delta=0.01))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        # 20 possible connections. Only those with a sufficiently small distance
        # are created
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123)])


@register_class()
class TestFromListConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    @register()
    def test_connect_with_valid_list(self, sim=sim):
        connection_list = [
            (0, 0, 0.1, 0.1),
            (3, 0, 0.2, 0.11),
            (2, 3, 0.3, 0.12),  # local
            (2, 2, 0.4, 0.13),
            (0, 1, 0.5, 0.14),  # local
            ]
        C = connectors.FromListConnector(connection_list)
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14),
                          (2, 3, 0.3, 0.12)])

    @register()
    def test_connect_with_out_of_range_index(self, sim=sim):
        connection_list = [
            (0, 0, 0.1, 0.1),
            (3, 0, 0.2, 0.11),
            (2, 3, 0.3, 0.12),  # local
            (5, 1, 0.4, 0.13),  # NON-EXISTENT
            (0, 1, 0.5, 0.14),  # local
            ]
        C = connectors.FromListConnector(connection_list)
        syn = sim.StaticSynapse()
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C, syn)

    @register()
    def test_with_plastic_synapse(self, sim=sim):
        connection_list = [
            (0, 0, 0.1, 0.1, 100, 400),
            (3, 0, 0.2, 0.11, 101, 500),
            (2, 3, 0.3, 0.12, 102, 600),  # local
            (2, 2, 0.4, 0.13, 103, 700),
            (0, 1, 0.5, 0.14, 104, 800),  # local
            ]
        C = connectors.FromListConnector(connection_list, column_names=["weight", "delay", "U", "tau_rec"])
        syn = sim.TsodyksMarkramSynapse(U=99, tau_facil=88.8)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay", "tau_facil", "tau_rec", "U"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14, 88.8, 800.0, 104.0),
                          (2, 3, 0.3, 0.12, 88.8, 600.0, 102.0)])


@register_class()
class TestFromFileConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        self.connection_list = [
            (0, 0, 0.1, 0.1),
            (3, 0, 0.2, 0.11),
            (2, 3, 0.3, 0.12),  # local
            (2, 2, 0.4, 0.13),
            (0, 1, 0.5, 0.14),  # local
            ]

    def tearDown(self, sim=sim):
        for path in ("test.connections", "test.connections.1", "test.connections.2"):
            if os.path.exists(path):
                os.remove(path)

    @register()
    def test_connect_with_standard_text_file_not_distributed(self, sim=sim):
        numpy.savetxt("test.connections", self.connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=False)
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14),
                          (2, 3, 0.3, 0.12)])

    @register()
    def test_connect_with_standard_text_file_distributed(self, sim=sim):
        local_connection_list = [c for c in self.connection_list if c[1]%2 == 1]
        numpy.savetxt("test.connections.1", local_connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=True)
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14),
                          (2, 3, 0.3, 0.12)])

    @register()
    def test_with_plastic_synapses_not_distributed(self, sim=sim):
        connection_list = [
            (0, 0, 0.1, 0.1,  100, 100),
            (3, 0, 0.2, 0.11, 110, 99),
            (2, 3, 0.3, 0.12, 120, 98),  # local
            (2, 2, 0.4, 0.13, 130, 97),
            (0, 1, 0.5, 0.14, 140, 96),  # local
            ]
        file = recording.files.StandardTextFile("test.connections.2", mode='wb')
        file.write(connection_list, {"columns": ["i", "j", "weight", "delay", "U", "tau_rec"]})
        C = connectors.FromFileConnector("test.connections.2", distributed=False)
        syn = sim.TsodyksMarkramSynapse(tau_facil=88.8)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay", "U", "tau_rec", "tau_facil"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14, 140.0, 96.0, 88.8),
                          (2, 3, 0.3, 0.12, 120.0, 98.0, 88.8)])


@register_class()
class TestFixedNumberPostConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    @register()
    def test_with_n_smaller_than_population_size(self, sim=sim):
        C = connectors.FixedNumberPostConnector(n=3, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse(weight="0.5*d")
        prj = sim.Projection(self.p1, self.p2, C, syn)
        # MockRNG.permutation(arr) returns the reverse of arr, so each pre neuron will connect to neurons 4, 3, 2
        # however, only neuron 3 is on the "local" (fake MPI) node
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 3, 1.5, 0.123),
                          (1, 3, 1.0, 0.123),
                          (2, 3, 0.5, 0.123),
                          (3, 3, 0.0, 0.123)])

    @register()
    def test_with_n_larger_than_population_size(self, sim=sim):
        C = connectors.FixedNumberPostConnector(n=7, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        # each pre neuron will connect to all post neurons (population size 5 is less than n), then to 4, 3 (MockRNG.permutation)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123)])

    @register()
    def test_with_n_larger_than_population_size_no_self_connections(self, sim=sim):
        C = connectors.FixedNumberPostConnector(n=7, allow_self_connections=False, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p2, self.p2, C, syn)
        # connections as follows: (pre - list of post)
        #   0 - 1 2 3 4 4 3 2
        #   1 - 0 2 3 4 4 3 2
        #   2 - 0 1 3 4 4 3 1
        #   3 - 0 1 2 4 4 2 1
        #   4 - 0 1 2 3 3 2 1
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (4, 1, 0.0, 0.123),
                          (4, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (4, 3, 0.0, 0.123),
                          (4, 3, 0.0, 0.123),])

    @register()
    def test_with_replacement(self, sim=sim):
        C = connectors.FixedNumberPostConnector(n=3, with_replacement=True, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        # 0 - 0 1 2
        # 1 - 3 4 0
        # 2 - 1 2 3
        # 3 - 4 0 1
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123)])
    @register()
    def test_with_replacement_with_variable_n(self, sim=sim):
        n = random.RandomDistribution('binomial', (5, 0.5), rng=MockRNG(start=1, delta=2))
            # should give (1, 3, 0, 2)
        C = connectors.FixedNumberPostConnector(n=n, with_replacement=True, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        # 0 - 0
        # 1 - 1 2 3
        # 2 -
        # 3 - 4 0
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 1, 0.0, 0.123),
                          (1, 3, 0.0, 0.123)])

    @register()
    def test_with_replacement_no_self_connections(self, sim=sim):
        C = connectors.FixedNumberPostConnector(n=3, with_replacement=True,
                                               allow_self_connections=False, rng=MockRNG(start=2, delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p2, self.p2, C, syn)
        # 0 - 2 3 4
        # 1 - 0 2 3
        # 2 - 4 0 1
        # 3 - 2 4 0
        # 4 - 1 2 3
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(2, 1, 0.0, 0.123),
                          (4, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (4, 3, 0.0, 0.123)])


@register_class()
class TestFixedNumberPreConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    @register()
    def test_with_n_smaller_than_population_size(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=3, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse(weight="0.1*d")
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(3, 1, 0.2, 0.123),
                          (2, 1, 0.1, 0.123),
                          (1, 1, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),
                          (2, 3, 0.1, 0.123),
                          (1, 3, 0.2, 0.123),])

    @register()
    def test_with_n_larger_than_population_size(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=7, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),])

    @register()
    def test_with_n_larger_than_population_size_no_self_connections(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=7, allow_self_connections=False, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p2, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (4, 1, 0.0, 0.123),
                          (4, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (4, 3, 0.0, 0.123),
                          (4, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),])

    @register()
    def test_with_replacement(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=3, with_replacement=True, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [#(0, 0, 0.0, 0.123),
                          #(1, 0, 0.0, 0.123),
                          #(2, 0, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          #(2, 2, 0.0, 0.123),
                          #(3, 2, 0.0, 0.123),
                          #(0, 2, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),])

    @register()
    def test_with_replacement_with_variable_n(self, sim=sim):
        n = random.RandomDistribution('binomial', (5, 0.5), rng=MockRNG(start=1, delta=2))
            # should give (1, 3, 0, 2, 4)
        C = connectors.FixedNumberPreConnector(n=n, with_replacement=True, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [#(0, 0, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123)])

    @register()
    def test_with_replacement_no_self_connections(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=3, with_replacement=True,
                                               allow_self_connections=False, rng=MockRNG(start=2, delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p2, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [#(2, 0, 0.0, 0.123),  # [2, 3, 4] --> [2, 3, 4]
                          #(3, 0, 0.0, 0.123),
                          #(4, 0, 0.0, 0.123),
                          (0, 1, 0.0, 0.123),   # [0, 1, 2] --> [0, 3, 2]
                          #(1, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          #(4, 2, 0.0, 0.123),  # [4, 0, 1] --> [4, 0, 1]
                          #(0, 2, 0.0, 0.123),
                          #(1, 2, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),   # [2, 3, 4] --> [2, 0, 4]
                          #(3, 3, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (4, 3, 0.0, 0.123),
                          ])

    @register()
    def test_no_replacement_no_self_connections(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=3, with_replacement=False,
                                               allow_self_connections=False, rng=MockRNG(start=2, delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p2, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(4, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (4, 3, 0.0, 0.123),
                          #(3, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),])

    @register()
    def test_with_replacement_parallel_unsafe(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=3, with_replacement=True, rng=MockRNG(delta=1, parallel_safe=False))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),
                          (0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),])

    @register()
    def test_no_replacement_parallel_unsafe(self, sim=sim):
        C = connectors.FixedNumberPreConnector(n=3, with_replacement=False, rng=MockRNG(delta=1, parallel_safe=False))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(3, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),])


@register_class()
class TestArrayConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(3, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(4, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([1,0,1,0], dtype=bool))

    @register()
    def test_connect_with_scalar_weights_and_delays(self, sim=sim):
        connections = numpy.array([
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ], dtype=bool)
        C = connectors.ArrayConnector(connections, safe=False)
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 0, 5.0, 0.5),
                          (0, 2, 5.0, 0.5),
                          (2, 2, 5.0, 0.5)])

    @register()
    def test_connect_with_random_weights_parallel_safe(self, sim=sim):
        rd_w = random.RandomDistribution('uniform', (0, 1), rng=MockRNG(delta=1.0, parallel_safe=True))
        rd_d = random.RandomDistribution('uniform', (0, 1), rng=MockRNG(start=1.0, delta=0.1, parallel_safe=True))
        syn = sim.StaticSynapse(weight=rd_w, delay=rd_d)
        connections = numpy.array([
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ], dtype=bool)
        C = connectors.ArrayConnector(connections, safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 0, 0.0, 1.0),
                          (0, 2, 3.0, 1.3),
                          (2, 2, 4.0, 1.4000000000000001)])  # better to do an "almost-equal" check



@register_class()
class TestCloneConnector(unittest.TestCase):

    def setUp(self, sim=sim):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        connection_list = [
            (0, 0, 0.0, 1.0),
            (3, 0, 0.0, 1.0),
            (2, 3, 0.0, 1.0),  # local
            (2, 2, 0.0, 1.0),
            (0, 1, 0.0, 1.0),  # local
            ]
        list_connector = connectors.FromListConnector(connection_list)
        syn = sim.StaticSynapse()
        self.ref_prj = sim.Projection(self.p1, self.p2, list_connector, syn)
        self.orig_gather_dict = recording.gather_dict  # create reference to original function
        # The gather_dict function in recording needs to be temporarily replaced so it can work with
        # a mock version of the function to avoid it throwing an mpi4py import error when setting
        # the rank in pyNN.mock by hand to > 1
        def mock_gather_dict(D, all=False):
            return D
        recording.gather_dict = mock_gather_dict

    def tearDown(self, sim=sim):
        # restore original gather_dict function
        recording.gather_dict = self.orig_gather_dict

    @register()
    def test_connect(self, sim=sim):
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        C = connectors.CloneConnector(self.ref_prj)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 5.0, 0.5),
                          (2, 3, 5.0, 0.5)])

    @register()
    def test_connect_with_pre_post_mismatch(self, sim=sim):
        syn = sim.StaticSynapse()
        C = connectors.CloneConnector(self.ref_prj)
        p3 = sim.Population(5, sim.IF_cond_exp(), structure=space.Line())
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, p3, C, syn)


@register_class()
class TestIndexBasedProbabilityConnector(unittest.TestCase):

    class IndexBasedProbability(connectors.IndexBasedExpression):
        def __call__(self, i, j):
            return numpy.array((i + j) % 3 == 0, dtype=float)

    class IndexBasedWeights(connectors.IndexBasedExpression):
        def __call__(self, i, j):
            return numpy.array(i * j + 1, dtype=float)

    class IndexBasedDelays(connectors.IndexBasedExpression):
        def __call__(self, i, j):
            return numpy.array(i + j + 1, dtype=float)

    def setUp(self, sim=sim, **extra):
        sim.setup(num_processes=2, rank=1, min_delay=0.123, **extra)
        self.p1 = sim.Population(5, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([1,0,1,0,1], dtype=bool))

    @register()
    def test_connect_with_scalar_weights_and_delays(self, sim=sim):
        syn = sim.StaticSynapse(weight=1.0, delay=2)
        C = connectors.IndexBasedProbabilityConnector(self.IndexBasedProbability())
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 0, 1, 2),
                          (3, 0, 1, 2),
                          (1, 2, 1, 2),
                          (4, 2, 1, 2),
                          (2, 4, 1, 2)])

    @register()
    def test_connect_with_index_based_weights(self, sim=sim):
        syn = sim.StaticSynapse(weight=self.IndexBasedWeights(), delay=2)
        C = connectors.IndexBasedProbabilityConnector(self.IndexBasedProbability())
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 0, 1, 2),
                          (3, 0, 1, 2),
                          (1, 2, 3, 2),
                          (4, 2, 9, 2),
                          (2, 4, 9, 2)])

    @register()
    def test_connect_with_index_based_delays(self, sim=sim):
        syn = sim.StaticSynapse(weight=1.0, delay=self.IndexBasedDelays())
        C = connectors.IndexBasedProbabilityConnector(self.IndexBasedProbability())
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 0, 1, 1),
                          (3, 0, 1, 4),
                          (1, 2, 1, 4),
                          (4, 2, 1, 7),
                          (2, 4, 1, 7)])


@register_class()
class TestDisplacementDependentProbabilityConnector(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        sim.setup(num_processes=2, rank=1, min_delay=0.123, **extra)
        self.p1 = sim.Population(9, sim.IF_cond_exp(),
                                 structure=space.Grid2D(aspect_ratio=1.0, dx=1.0, dy=1.0))
        self.p2 = sim.Population(9, sim.HH_cond_exp(),
                                 structure=space.Grid2D(aspect_ratio=1.0, dx=1.0, dy=1.0))
        assert_array_equal(self.p2._mask_local, numpy.array([1,0,1,0,1,0,1,0,1], dtype=bool))

    @register()
    def test_connect(self, sim=sim):
        syn = sim.StaticSynapse(weight=1.0, delay=2)
        def displacement_expression(d):
            return 0.5 * ((d[0] >= -1) * (d[0] <= 2)) + 0.25 * (d[1] >= 0) * (d[1] <= 1)
        C = connectors.DisplacementDependentProbabilityConnector(displacement_expression,
                                                                 rng=MockRNG(delta=0.01))
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 0, 1.0, 2.0),
                          (1, 0, 1.0, 2.0),
                          (2, 0, 1.0, 2.0),
                          (3, 0, 1.0, 2.0),
                          (4, 0, 1.0, 2.0),
                          (5, 0, 1.0, 2.0),
                          (6, 0, 1.0, 2.0),
                          (0, 2, 1.0, 2.0),
                          (1, 2, 1.0, 2.0),
                          (2, 2, 1.0, 2.0),
                          (3, 2, 1.0, 2.0),
                          (4, 2, 1.0, 2.0),
                          (5, 2, 1.0, 2.0),
                          (0, 4, 1.0, 2.0),
                          (1, 4, 1.0, 2.0),
                          (2, 4, 1.0, 2.0),
                          (3, 4, 1.0, 2.0),
                          (4, 4, 1.0, 2.0),
                          (5, 4, 1.0, 2.0),
                          (6, 4, 1.0, 2.0),
                          (7, 4, 1.0, 2.0),
                          (8, 4, 1.0, 2.0),
                          (0, 6, 1.0, 2.0),
                          (3, 6, 1.0, 2.0),
                          (6, 6, 1.0, 2.0),
                          (1, 8, 1.0, 2.0),
                          (2, 8, 1.0, 2.0)])

@unittest.skip('skipping these tests until I figure out how I want to refactor checks')
class CheckTest(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        self.MIN_DELAY = 0.123
        sim.setup(num_processes=2, rank=1, min_delay=0.123, **extra)

    def test_check_weights_with_scalar(self, sim=sim):
        self.assertEqual(4.3, connectors.check_weights(4.3, 'excitatory', is_conductance=True))
        self.assertEqual(4.3, connectors.check_weights(4.3, 'excitatory', is_conductance=False))
        self.assertEqual(4.3, connectors.check_weights(4.3, 'inhibitory', is_conductance=True))
        self.assertEqual(-4.3, connectors.check_weights(-4.3, 'inhibitory', is_conductance=False))
        self.assertEqual(connectors.DEFAULT_WEIGHT, connectors.check_weights(None, 'excitatory', is_conductance=True))
        self.assertRaises(errors.ConnectionError, connectors.check_weights, 4.3, 'inhibitory', is_conductance=False)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, -4.3, 'inhibitory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, -4.3, 'excitatory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, -4.3, 'excitatory', is_conductance=False)

    def test_check_weights_with_array(self, sim=sim):
        w = numpy.arange(10)
        assert_array_equal(w, connectors.check_weights(w, 'excitatory', is_conductance=True))
        assert_array_equal(w, connectors.check_weights(w, 'excitatory', is_conductance=False))
        assert_array_equal(w, connectors.check_weights(w, 'inhibitory', is_conductance=True))
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'inhibitory', is_conductance=False)
        w = numpy.arange(-10,0)
        assert_array_equal(w, connectors.check_weights(w, 'inhibitory', is_conductance=False))
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'inhibitory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'excitatory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'excitatory', is_conductance=False)
        w = numpy.arange(-5,5)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'excitatory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'excitatory', is_conductance=False)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'inhibitory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, w, 'inhibitory', is_conductance=False)

    def test_check_weights_with_invalid_value(self, sim=sim):
        self.assertRaises(errors.ConnectionError, connectors.check_weights, "butterflies", 'excitatory', is_conductance=True)

    def test_check_weight_is_conductance_is_None(self, sim=sim):
        # need to check that a log message was created
        self.assertEqual(4.3, connectors.check_weights(4.3, 'excitatory', is_conductance=None))

    def test_check_delay(self, sim=sim):
        self.assertEqual(connectors.check_delays(2*self.MIN_DELAY, self.MIN_DELAY, 1e99), 2*self.MIN_DELAY)
        self.assertRaises(errors.ConnectionError, connectors.check_delays, 0.5*self.MIN_DELAY, self.MIN_DELAY, 1e99)
        self.assertRaises(errors.ConnectionError, connectors.check_delays, 3.0, self.MIN_DELAY, 2.0)
