"""
Tests of the Connector classes, using the pyNN.mock backend.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from pyNN import connectors, random, errors, space, recording
import numpy
import os
from mock import Mock
from numpy.testing import assert_array_equal, assert_array_almost_equal
from .mocks import MockRNG
import pyNN.mock as sim


orig_mpi_get_config = random.get_mpi_config
def setUp():
    random.get_mpi_config = lambda: (0, 2)

def tearDown():
    random.get_mpi_config = orig_mpi_get_config

class TestOneToOneConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=0)
        self.p1 = sim.Population(5, sim.IF_cond_exp())
        self.p2 = sim.Population(5, sim.HH_cond_exp())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_connect_with_scalar_weights_and_delays(self):
        C = connectors.OneToOneConnector(safe=False)
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 1, 5.0, 0.5),
                          (3, 3, 5.0, 0.5)])

    def test_connect_with_random_weights(self):
        rd = random.RandomDistribution(rng=MockRNG(delta=1.0))
        syn = sim.StaticSynapse(weight=rd, delay=0.5)
        C = connectors.OneToOneConnector(safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 1, 1.0, 0.5),
                          (3, 3, 3.0, 0.5)])


class TestAllToAllConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_connect_with_scalar_weights_and_delays(self):
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

    def test_connect_with_random_weights_parallel_safe(self):
        rd = random.RandomDistribution(rng=MockRNG(delta=1.0, parallel_safe=True))
        syn = sim.StaticSynapse(weight=rd, delay=0.5)
        C = connectors.AllToAllConnector(safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 4.0, 0.5),
                          (1, 1, 5.0, 0.5),
                          (2, 1, 6.0, 0.5),
                          (3, 1, 7.0, 0.5),
                          (0, 3, 12.0, 0.5),
                          (1, 3, 13.0, 0.5),
                          (2, 3, 14.0, 0.5),
                          (3, 3, 15.0, 0.5)])

    def test_connect_with_distance_dependent_weights(self):
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

    def test_connect_with_delays_None(self):
        syn = sim.StaticSynapse(weight=0.1, delay=None)
        C = connectors.AllToAllConnector()
        assert C.safe
        assert C.allow_self_connections
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False)[0][3], prj._simulator.state.min_delay)

    @unittest.skip('skipping this tests until I figure out how I want to refactor checks')
    def test_connect_with_delays_too_small(self):
        C = connectors.AllToAllConnector()
        syn = sim.StaticSynapse(weight=0.1, delay=0.0)
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C, syn)

    @unittest.skip('skipping this tests until I figure out how I want to refactor checks')
    def test_connect_with_list_delays_too_small(self):
        delays = numpy.ones((self.p1.size, self.p2.size), float)
        delays[2, 3] = sim.Projection._simulator.state.min_delay - 0.01
        syn = sim.StaticSynapse(weight=0.1, delay=delays)
        C = connectors.AllToAllConnector()
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, self.p2, C, syn)


class TestFixedProbabilityConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_connect_with_default_args(self):
        C = connectors.FixedProbabilityConnector(p_connect=0.75,
                                                 rng=MockRNG(delta=0.1, parallel_safe=True))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)

        # 20 possible connections. Due to the mock RNG, only the
        # first 8 are created (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)
        # of these, (0,1), (1,1), (2,1), (3,1) are created on this node
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123)])

    def test_connect_with_weight_function(self):
        C = connectors.FixedProbabilityConnector(p_connect=0.75,
                                                 rng=MockRNG(delta=0.1))
        syn = sim.StaticSynapse(weight=lambda d: 0.1*d)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.1, 0.123),
                          (1, 1, 0.0, 0.123),
                          (2, 1, 0.1, 0.123),
                          (3, 1, 0.2, 0.123)])

    def test_connect_with_random_delays_parallel_safe(self):
        rd = random.RandomDistribution('uniform', [0.1, 1.1], rng=MockRNG(start=1.0, delta=0.2, parallel_safe=True))
        syn = sim.StaticSynapse(delay=rd)
        C = connectors.FixedProbabilityConnector(p_connect=0.75, rng=MockRNG(delta=0.1))
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 1.0+0.2*4),
                          (1, 1, 0.0, 1.0+0.2*5),
                          (2, 1, 0.0, 1.0+0.2*6),
                          (3, 1, 0.0, 1.0+0.2*7)])


class TestDistanceDependentProbabilityConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_connect_with_default_args(self):
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


class TestFromListConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_connect_with_valid_list(self):
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

    def test_connect_with_out_of_range_index(self):
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


class TestCloneConnector(unittest.TestCase):

    def setUp(self):
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
        
    def tearDown(self):
        # restore original gather_dict function
        recording.gather_dict = self.orig_gather_dict  

    def test_connect_with_scalar_weights_and_delays(self):
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        C = connectors.CloneConnector(self.ref_prj)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 5.0, 0.5),
                          (2, 3, 5.0, 0.5)])

    def test_connect_with_random_weights_parallel_safe(self):
        rd = random.RandomDistribution(rng=MockRNG(delta=1.0, parallel_safe=True))
        syn = sim.StaticSynapse(weight=rd, delay=0.5)
        C = connectors.CloneConnector(self.ref_prj)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.0, 0.5),
                          (2, 3, 1.0, 0.5)])
        
    def test_connect_with_distance_dependent_weights(self):
        d_expr = "d+100"
        syn = sim.StaticSynapse(weight=d_expr, delay=0.5)
        C = connectors.CloneConnector(self.ref_prj)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 101.0, 0.5),
                          (2, 3, 101.0, 0.5)])

    def test_connect_with_pre_post_mismatch(self):
        syn = sim.StaticSynapse()
        C = connectors.CloneConnector(self.ref_prj)
        p3 = sim.Population(5, sim.IF_cond_exp(), structure=space.Line())
        self.assertRaises(errors.ConnectionError, sim.Projection, self.p1, p3, C, syn)


class TestFromFileConnector(unittest.TestCase):

    def setUp(self):
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

    def tearDown(self):
        for path in ("test.connections", "test.connections.1"):
            if os.path.exists(path):
                os.remove(path)

    def test_connect_with_standard_text_file_not_distributed(self):
        numpy.savetxt("test.connections", self.connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=False)
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14),
                          (2, 3, 0.3, 0.12)])

    def test_connect_with_standard_text_file_distributed(self):
        local_connection_list = [c for c in self.connection_list if c[1]%2 == 1]
        numpy.savetxt("test.connections.1", local_connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=True)
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 1, 0.5, 0.14),
                          (2, 3, 0.3, 0.12)])


class TestFixedNumberPostConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_with_n_smaller_than_population_size(self):
        C = connectors.FixedNumberPostConnector(n=3, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(0, 3, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),])


class TestFixedNumberPreConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(4, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(5, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([0,1,0,1,0], dtype=bool))

    def test_with_n_smaller_than_population_size(self):
        C = connectors.FixedNumberPreConnector(n=3, rng=MockRNG(delta=1))
        syn = sim.StaticSynapse()
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 1, 0.0, 0.123),
                          (2, 1, 0.0, 0.123),
                          (3, 1, 0.0, 0.123),
                          (1, 3, 0.0, 0.123),
                          (2, 3, 0.0, 0.123),
                          (3, 3, 0.0, 0.123),])


class TestArrayConnector(unittest.TestCase):

    def setUp(self):
        sim.setup(num_processes=2, rank=1, min_delay=0.123)
        self.p1 = sim.Population(3, sim.IF_cond_exp(), structure=space.Line())
        self.p2 = sim.Population(4, sim.HH_cond_exp(), structure=space.Line())
        assert_array_equal(self.p2._mask_local, numpy.array([1,0,1,0], dtype=bool))

    def test_connect_with_scalar_weights_and_delays(self):
        connections = numpy.array([
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ])
        C = connectors.ArrayConnector(connections, safe=False)
        syn = sim.StaticSynapse(weight=5.0, delay=0.5)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 0, 5.0, 0.5),
                          (0, 2, 5.0, 0.5),
                          (2, 2, 5.0, 0.5)])

    def test_connect_with_random_weights_parallel_safe(self):
        rd_w = random.RandomDistribution(rng=MockRNG(delta=1.0, parallel_safe=True))
        rd_d = random.RandomDistribution(rng=MockRNG(start=1.0, delta=0.1, parallel_safe=True))
        syn = sim.StaticSynapse(weight=rd_w, delay=rd_d)
        connections = numpy.array([
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ])
        C = connectors.ArrayConnector(connections, safe=False)
        prj = sim.Projection(self.p1, self.p2, C, syn)
        self.assertEqual(prj.get(["weight", "delay"], format='list', gather=False),  # use gather False because we are faking the MPI
                         [(1, 0, 0.0, 1.0),
                          (0, 2, 3.0, 1.3),
                          (2, 2, 4.0, 1.4000000000000001)])  # better to do an "almost-equal" check
  

@unittest.skip('skipping these tests until I figure out how I want to refactor checks')
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
