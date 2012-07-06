"""
Tests of the Connector classes, using the pyNN.mock backend.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from pyNN import connectors, random, errors, space
import numpy
import os
from mock import Mock
from numpy.testing import assert_array_equal, assert_array_almost_equal
from .mocks import MockRNG
import pyNN.mock as sim

real_mpi_rank = random.mpi_rank
real_num_processes = random.num_processes
orig_sim_mpi_rank = sim.simulator.state.mpi_rank
orig_sim_num_processes = sim.simulator.state.num_processes

def tearDown():
    random.mpi_rank = real_mpi_rank
    random.num_processes = real_num_processes
    sim.simulator.state.mpi_rank = orig_sim_mpi_rank
    sim.simulator.state.num_processes = orig_sim_num_processes

class TestOneToOneConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=0)
        self.p1 = sim.Population(5, sim.IF_cond_exp())
        self.p2 = sim.Population(5, sim.HH_cond_exp())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_connect_with_scalar_weights_and_delays(self):
        C = connectors.OneToOneConnector(weights=5.0, delays=0.5, safe=False)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[1], self.p2[1], 5.0, 0.5),
                          (self.p1[3], self.p2[3], 5.0, 0.5)])

    def test_connect_with_random_weights(self):
        rd = random.RandomDistribution(rng=MockRNG(delta=1.0))
        C = connectors.OneToOneConnector(weights=rd, delays=0.5, safe=False)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[1], self.p2[1], 1.0, 0.5),
                          (self.p1[3], self.p2[3], 3.0, 0.5)])


class TestAllToAllConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2

    def test_connect_with_scalar_weights_and_delays(self):
        C = connectors.AllToAllConnector(weights=5.0, delays=0.5, safe=False)
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=0.1, parallel_safe=True))
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 5.0, 0.5),
                          (self.p1[1], self.p2[1], 5.0, 0.5),
                          (self.p1[2], self.p2[1], 5.0, 0.5),
                          (self.p1[3], self.p2[1], 5.0, 0.5),
                          (self.p1[0], self.p2[3], 5.0, 0.5),
                          (self.p1[1], self.p2[3], 5.0, 0.5),
                          (self.p1[2], self.p2[3], 5.0, 0.5),
                          (self.p1[3], self.p2[3], 5.0, 0.5)])

    def test_connect_with_random_weights_parallel_safe(self):
        rd = random.RandomDistribution(rng=MockRNG(delta=1.0, parallel_safe=True))
        C = connectors.AllToAllConnector(weights=rd, delays=0.5, safe=False)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 4.0, 0.5),
                          (self.p1[1], self.p2[1], 5.0, 0.5),
                          (self.p1[2], self.p2[1], 6.0, 0.5),
                          (self.p1[3], self.p2[1], 7.0, 0.5),
                          (self.p1[0], self.p2[3], 12.0, 0.5),
                          (self.p1[1], self.p2[3], 13.0, 0.5),
                          (self.p1[2], self.p2[3], 14.0, 0.5),
                          (self.p1[3], self.p2[3], 15.0, 0.5)])

    def test_connect_with_distance_dependent_weights_parallel_safe(self):
        d_expr = "d+100"
        C = connectors.AllToAllConnector(weights=d_expr, delays=0.5, safe=False)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 101.0, 0.5),
                          (self.p1[1], self.p2[1], 100.0, 0.5),
                          (self.p1[2], self.p2[1], 101.0, 0.5),
                          (self.p1[3], self.p2[1], 102.0, 0.5),
                          (self.p1[0], self.p2[3], 103.0, 0.5),
                          (self.p1[1], self.p2[3], 102.0, 0.5),
                          (self.p1[2], self.p2[3], 101.0, 0.5),
                          (self.p1[3], self.p2[3], 100.0, 0.5)])

    def test_connect_with_delays_None(self):
        C = connectors.AllToAllConnector(weights=0.1, delays=None)
        assert C.safe
        assert C.allow_self_connections
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections[0][3], prj._simulator.state.min_delay)

    def test_connect_with_delays_too_small(self):
        C = connectors.AllToAllConnector(weights=0.1, delays=0.0)
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C)

    def test_connect_with_list_delays_too_small(self):
        delays = numpy.ones((self.p1.size, self.p2.size), float)
        delays[2, 3] = sim.Projection._simulator.state.min_delay - 0.01
        C = connectors.AllToAllConnector(weights=0.1, delays=delays)
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C)


class TestFixedProbabilityConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2

    def test_connect_with_default_args(self):
        C = connectors.FixedProbabilityConnector(p_connect=0.75)
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=0.1, parallel_safe=True))

        # 20 possible connections. Due to the mock RNG, only the
        # first 8 are created (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)
        # of these, (0,1), (1,1), (2,1), (3,1) are created on this node
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.0, 0.123),
                          (self.p1[1], self.p2[1], 0.0, 0.123),
                          (self.p1[2], self.p2[1], 0.0, 0.123),
                          (self.p1[3], self.p2[1], 0.0, 0.123)])

    def test_connect_with_weight_function(self):
        C = connectors.FixedProbabilityConnector(p_connect=0.75, weights=lambda d: 0.1*d)
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=0.1))
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.1, 0.123),
                          (self.p1[1], self.p2[1], 0.0, 0.123),
                          (self.p1[2], self.p2[1], 0.1, 0.123),
                          (self.p1[3], self.p2[1], 0.2, 0.123)])

    def test_connect_with_random_delays_parallel_safe(self):
        rd = random.RandomDistribution('uniform', [0.1, 1.1], rng=MockRNG(start=1.0, delta=0.2, parallel_safe=True))
        C = connectors.FixedProbabilityConnector(p_connect=0.75, delays=rd)
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=0.1))
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.0, 1.0+0.2*4),
                          (self.p1[1], self.p2[1], 0.0, 1.0+0.2*5),
                          (self.p1[2], self.p2[1], 0.0, 1.0+0.2*6),
                          (self.p1[3], self.p2[1], 0.0, 1.0+0.2*7)])


class TestDistanceDependentProbabilityConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2

    def test_connect_with_default_args(self):
        C = connectors.DistanceDependentProbabilityConnector(d_expression="d<1.5")
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=0.01))
        # 20 possible connections. Only those with a sufficiently small distance
        # are created
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.0, 0.123),
                          (self.p1[1], self.p2[1], 0.0, 0.123),
                          (self.p1[2], self.p2[1], 0.0, 0.123),
                          (self.p1[2], self.p2[3], 0.0, 0.123),
                          (self.p1[3], self.p2[3], 0.0, 0.123)])


class TestFromListConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2

    def test_connect_with_valid_list(self):
        connection_list = [
            (0, 0, 0.1, 0.1),
            (3, 0, 0.2, 0.11),
            (2, 3, 0.3, 0.12),  # local
            (2, 2, 0.4, 0.13),
            (0, 1, 0.5, 0.14),  # local
            ]
        C = connectors.FromListConnector(connection_list)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.5, 0.14),
                          (self.p1[2], self.p2[3], 0.3, 0.12)])

    def test_connect_with_out_of_range_index(self):
        connection_list = [
            (0, 0, 0.1, 0.1),
            (3, 0, 0.2, 0.11),
            (2, 3, 0.3, 0.12),  # local
            (5, 1, 0.4, 0.13),  # NON-EXISTENT
            (0, 1, 0.5, 0.14),  # local
            ]
        C = connectors.FromListConnector(connection_list)
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C)


class TestFromFileConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2
        self.connection_list = [
            (0, 0, 0.1, 0.1),
            (3, 0, 0.2, 0.11),
            (2, 3, 0.3, 0.12),  # local
            (2, 2, 0.4, 0.13),
            (0, 1, 0.5, 0.14),  # local
            ]

    def tearDown(self):
        for path in ("test.connections", "test.connections.1"):
            if os.path.exists(path):
                os.remove(path)

    def test_connect_with_standard_text_file_not_distributed(self):
        numpy.savetxt("test.connections", self.connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=False)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.5, 0.14),
                          (self.p1[2], self.p2[3], 0.3, 0.12)])

    def test_connect_with_standard_text_file_distributed(self):
        local_connection_list = [c for c in self.connection_list if c[1]%2 == 1]
        numpy.savetxt("test.connections.1", local_connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=True)
        prj = sim.Projection(self.p1, self.p2, C)
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[1], 0.5, 0.14),
                          (self.p1[2], self.p2[3], 0.3, 0.12)])


class TestFixedNumberPostConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2

    def test_with_n_smaller_than_population_size(self):
        C = connectors.FixedNumberPostConnector(n=3)
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=1))
        self.assertEqual(prj.connections,
                         [(self.p1[0], self.p2[3], 0.0, 0.123),
                          (self.p1[1], self.p2[3], 0.0, 0.123),
                          (self.p1[2], self.p2[3], 0.0, 0.123),
                          (self.p1[3], self.p2[3], 0.0, 0.123),])


class TestFixedNumberPreConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))
        random.mpi_rank = 1
        random.num_processes = 2

    def test_with_n_smaller_than_population_size(self):
        C = connectors.FixedNumberPreConnector(n=3)
        prj = sim.Projection(self.p1, self.p2, C, rng=MockRNG(delta=1))
        self.assertEqual(prj.connections,
                         [(self.p1[1], self.p2[1], 0.0, 0.123),
                          (self.p1[2], self.p2[1], 0.0, 0.123),
                          (self.p1[3], self.p2[1], 0.0, 0.123),
                          (self.p1[1], self.p2[3], 0.0, 0.123),
                          (self.p1[2], self.p2[3], 0.0, 0.123),
                          (self.p1[3], self.p2[3], 0.0, 0.123),])


class CheckTest(unittest.TestCase):

    def setUp(self):
        self.MIN_DELAY = 0.123
        sim.setup(num_processes=2, rank=1, min_delay=0.123)

    def test_check_weights_with_scalar(self):
        self.assertEqual(4.3, connectors.check_weights(4.3, 'excitatory', is_conductance=True))
        self.assertEqual(4.3, connectors.check_weights(4.3, 'excitatory', is_conductance=False))
        self.assertEqual(4.3, connectors.check_weights(4.3, 'inhibitory', is_conductance=True))
        self.assertEqual(-4.3, connectors.check_weights(-4.3, 'inhibitory', is_conductance=False))
        self.assertEqual(connectors.DEFAULT_WEIGHT, connectors.check_weights(None, 'excitatory', is_conductance=True))
        self.assertRaises(errors.ConnectionError, connectors.check_weights, 4.3, 'inhibitory', is_conductance=False)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, -4.3, 'inhibitory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, -4.3, 'excitatory', is_conductance=True)
        self.assertRaises(errors.ConnectionError, connectors.check_weights, -4.3, 'excitatory', is_conductance=False)

    def test_check_weights_with_array(self):
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

    def test_check_weights_with_invalid_value(self):
        self.assertRaises(errors.ConnectionError, connectors.check_weights, "butterflies", 'excitatory', is_conductance=True)

    def test_check_weight_is_conductance_is_None(self):
        # need to check that a log message was created
        self.assertEqual(4.3, connectors.check_weights(4.3, 'excitatory', is_conductance=None))

    def test_check_delay(self):
        self.assertEqual(connectors.check_delays(2*self.MIN_DELAY, self.MIN_DELAY, 1e99), 2*self.MIN_DELAY)
        self.assertRaises(errors.ConnectionError, connectors.check_delays, 0.5*self.MIN_DELAY, self.MIN_DELAY, 1e99)
        self.assertRaises(errors.ConnectionError, connectors.check_delays, 3.0, self.MIN_DELAY, 2.0)
