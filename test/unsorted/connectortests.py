from mpi4py import MPI
import numpy
from pyNN.random import AbstractRNG, RandomDistribution
from pyNN.connectors import FixedProbabilityConnector, AllToAllConnector, DistanceMatrix, \
                            ConnectionAttributeGenerator
from pyNN import common, errors
from pyNN.space import Space
import unittest

mpi_comm = MPI.COMM_WORLD

def assert_arrays_almost_equal(a, b, threshold, msg=''):
    if a.shape != b.shape:
        raise unittest.TestCase.failureException("Shape mismatch: a.shape=%s, b.shape=%s" % (a.shape, b.shape))
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        if msg:
            err_msg += "\nOther information: %s" % msg
        raise unittest.TestCase.failureException(err_msg)

class MockSimulatorModule(object):
    class State(object):
        min_delay = 0.1
        max_delay = 10.0
    state = State()

common.simulator = MockSimulatorModule()

class MockRNG(AbstractRNG):
    
    def __init__(self, seed):
        self.current = seed
        
    def next(self, n=1, distribution='uniform', parameters=[], mask_local=None):
        start = self.current
        self.current += n
        return numpy.arange(start, self.current)    

class MockConnectionManager(object):
        def __init__(self):
            self.n = 0
            self.weights = []
            self.delays = []
            
        def connect(self, src, targets, weights, delays):
            assert len(targets) == len(weights) == len(delays) > 0, "%d %d %d" % (len(targets), len(weights), len(delays))
            self.n += len(targets)
            self.weights.extend(weights)
            self.delays.extend(delays)
            
class MockProjection(object):
    
    def __init__(self, pre, post, rng, synapse_type):
        self.pre = pre
        self.post = post
        self.rng = rng
        self.synapse_type = synapse_type
        self.connection_manager = MockConnectionManager()

    @property
    def n(self):
        return self.connection_manager.n

class MockID(int):
    
    def __init__(self, id):
        self.id = id
    
    @property
    def position(self):
        return self.parent.positions.T[self.id]        

class MockPopulation(object):
    
    def __init__(self, n):
        self.all_cells = numpy.array([MockID(i) for i in range(n)], dtype=MockID)
        for i in range(n):
            self.all_cells[i].parent = self
        self.positions = numpy.array([(i, 0.0, 0.0) for i in self.all_cells], dtype=float).T
        self._mask_local = numpy.array([i%mpi_comm.size == mpi_comm.rank for i in range(n)])
        self.local_cells = self.all_cells[self._mask_local]
    
    def all(self):
        return self.all_cells
    
    @property
    def size(self):
        return self.all_cells.size
    
    def __len__(self):
        return self.all_cells.size
        

class AllToAllConnectorTest(unittest.TestCase):
    
    def setUp(self):
        self.p1 = MockPopulation(17)
        self.p2 = MockPopulation(13)
        self.rng = MockRNG(0)
        
    
    def test_create_with_delays_None(self):
        connector = AllToAllConnector(weights=0.1, delays=None)
        self.assertEqual(connector.weights, 0.1)
        self.assertEqual(connector.delays, common.get_min_delay())
        self.assert_(connector.safe)
        self.assert_(connector.allow_self_connections)
        
    def test_create_with_delays_too_small(self):
        self.assertRaises(errors.ConnectionError,
                          AllToAllConnector, allow_self_connections=True,
                          delays=0.0)

    def test_create_with_list_delays_too_small(self):
        self.assertRaises(errors.ConnectionError,
                          AllToAllConnector, allow_self_connections=True,
                          delays=[1.0, 1.0, 0.0])

    def test_connect_with_single_weight(self):
        connector = AllToAllConnector(allow_self_connections=True,
                                      weights=0.1)
        prj = MockProjection(self.p1, self.p2, self.rng, "excitatory")
        connector.connect(prj)
        self.assertEqual(prj.n, self.p1.size*self.p2.size)
        self.assertEqual(prj.connection_manager.weights, [[0.1]]*prj.n)
        
    def test_connect_with_no_self_connections(self):
        connector = AllToAllConnector(allow_self_connections=False,
                                      weights=0.1)
        prj = MockProjection(self.p1, self.p1, self.rng, "excitatory")
        connector.connect(prj)
        self.assertEqual(prj.n, self.p1.size*(self.p1.size-1))
        self.assertEqual(prj.connection_manager.weights, [[0.1]]*prj.n)
        
    def test_connect_with_random_weights(self):
        connector = AllToAllConnector(weights=RandomDistribution("uniform", [0.3, 0.4], rng=self.rng))
        prj = MockProjection(self.p1, self.p2, self.rng, "excitatory")
        connector.connect(prj)
        self.assertEqual(prj.n, self.p1.size*self.p2.size)
        assert_arrays_almost_equal(numpy.array(prj.connection_manager.weights), numpy.arange(0, prj.n), 1e-6)

    def test_connect_with_weight_array(self):
        w_in = numpy.arange(self.p1.size*self.p2.size, 0.0, -1.0).reshape(self.p1.size, self.p2.size)
        connector = AllToAllConnector(weights=w_in)
        prj = MockProjection(self.p1, self.p2, self.rng, "excitatory")
        connector.connect(prj)
        self.assertEqual(prj.n, self.p1.size*self.p2.size)
        assert_arrays_almost_equal(numpy.array(prj.connection_manager.weights), w_in.flatten(), 1e-6)

    def test_connect_with_distance_dependent_weights_simple(self):
        self.p1.positions = numpy.zeros((3,self.p1.size))
        connector = AllToAllConnector(weights="d*d")
        prj = MockProjection(self.p1, self.p2, self.rng, "excitatory")
        connector.connect(prj)
        self.assertEqual(prj.n, self.p1.size*self.p2.size)
        w_expected = numpy.array([w*w for w in range(self.p2.size)]*self.p1.size, float).flatten()
        assert_arrays_almost_equal(numpy.array(prj.connection_manager.weights),
                                   w_expected, 1e-6)

    def test_connect_with_distance_dependent_weights(self):
        # 3D position values, not just zeros and ones
        self.fail()

    def test_connect_with_distance_dependent_weights_and_delays(self):
        # check how many times DistanceMatrix.as_array() gets called
        self.fail()


class FixedProbabilityConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()
        
class DistanceDependentProbabilityConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class FixedNumberPreConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()
        
class FixedNumberPostConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class OneToOneConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class SmallWorldConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class CSAConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class FromListConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class FromFileConnectorTest(unittest.TestCase):

    def test(self):
        self.fail()

class TestDistanceMatrix(unittest.TestCase):
    
    def test_really_simple0(self):
        A = numpy.zeros((3,))
        B = numpy.zeros((3,5))
        D = DistanceMatrix(B, Space())
        D.set_source(A)
        assert_arrays_almost_equal(D.as_array(),
                                   numpy.zeros((5,), float),
                                   1e-12)

    def test_really_simple1(self):
        A = numpy.ones((3,))
        B = numpy.zeros((3,5))
        D = DistanceMatrix(B, Space())
        D.set_source(A)
        assert_arrays_almost_equal(D.as_array(),
                                   numpy.sqrt(3*numpy.ones((5,), float)),
                                   1e-12)


class TestConnectionAttributeGenerator(unittest.TestCase):
    
    def test_extract_with_simple_d_expr(self):
        B = numpy.zeros((3,5))
        dist = DistanceMatrix(B, Space())
        gen = ConnectionAttributeGenerator(
                  source="d*d",
                  local_mask=None,
                  safe=False)
        dist.set_source(numpy.zeros((3,)))
        assert_arrays_almost_equal(gen.extract(5, dist, sub_mask=None),
                                   numpy.zeros((5,), float),
                                   1e-12)
        dist.set_source(numpy.array([1,0,0]))
        assert_arrays_almost_equal(gen.extract(5, dist, sub_mask=None),
                                   numpy.ones((5,), float),
                                   1e-12)
        dist.set_source(numpy.array([2,0,0]))
        assert_arrays_almost_equal(gen.extract(5, dist, sub_mask=None),
                                   4*numpy.ones((5,), float),
                                   1e-12)

    
# ==============================================================================
if __name__ == "__main__":
    unittest.main()