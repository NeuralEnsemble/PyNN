from pyNN import connectors, common, random, errors, space
import numpy
import os
from mock import Mock
from nose.tools import assert_equal, assert_raises
from pyNN.utility import assert_arrays_equal
from itertools import repeat

MIN_DELAY = 0.123
MAX_DELAY = 99999

class MockSimulator(object):
    class MockState(object):
        min_delay = MIN_DELAY
        max_delay = MAX_DELAY
        num_processes = 2
        mpi_rank = 1
    state = MockState()

class MockCell(int):

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        self.position = numpy.array([n, 88.8, 99.9])

class MockPre(object):
    
    def __init__(self, size):
        self.size = size
        self.all_cells = numpy.arange(17, 17+size) # the 17 is just to make sure we make no assumptions about ID values starting at zero
        self.positions = numpy.array([(i, 88.8, 99.9) for i in self.all_cells]).T
        self.position_generator = lambda i: self.positions[:,i]
    
    def __len__(self):
        return self.size
    
    def all(self):
        return iter(MockCell(id) for id in self.all_cells)

class MockPost(object):
    
    def __init__(self, local_mask):
        self._mask_local = local_mask
        self.size = local_mask.size # can local mask be an array of indices or a slice?
        self.all_cells = numpy.arange(79, 79+self.size)
        self.local_cells = self.all_cells[local_mask]
        self.positions = numpy.array([(i, 88.8, 99.9) for i in self.all_cells]).T
        self.position_generator = lambda i: self.positions[:,i]

    def __len__(self):
        return self.size


class MockRNG(random.WrappedRNG):
    rng = None
    
    def __init__(self, num_processes, delta=1):
        random.num_processes = num_processes
        random.WrappedRNG.__init__(self)
        self.start = 0.0
        self.delta = delta
    
    def _next(self, distribution, n, parameters):
        s = self.start
        self.start += n*self.delta
        return numpy.arange(s, s+n*self.delta, self.delta)


class MockProjection(object):
    _simulator = MockSimulator
    
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        self.connections = []
        self.rng = MockRNG(num_processes=2, delta=0.1)
        self.synapse_type = 'inhibitory'

    def _divergent_connect(self, src, targets, weights, delays):
        #if src in self.connections:
        #    raise Exception("connect already called with source %s" % src) # no reason why this shouldn't happen, but it doesn't in the current implementation, so I'm being lazy
        #else:
        #    self.connections[src] = {"targets": targets,
        #                               "weights": weights,
        #                               "delays": delays}
        if isinstance(weights, float):
            weights = repeat(weights)
        if isinstance(delays, float):
            delays = repeat(delays)
        if not hasattr(targets, "__len__"):
            targets = [targets]
        for tgt, w, d in zip(targets, weights, delays):
            self.connections.append((src, tgt, w, d))

    def _convergent_connect(self, sources, tgt, weights, delays):
        if isinstance(weights, float):
            weights = repeat(weights)
        if isinstance(delays, float):
            delays = repeat(delays)
        for src, w, d in zip(sources, weights, delays):
            self.connections.append((src, tgt, w, d))


def setup():
    connectors.Connector._simulator = MockSimulator
    connectors.ConnectionAttributeGenerator._simulator = MockSimulator

def teardown():
    del connectors.Connector._simulator
    del connectors.ConnectionAttributeGenerator._simulator

class TestOneToOneConnector(object):

    def setup(self):
        self.prj = MockProjection(MockPre(5), MockPost(numpy.array([0,1,0,1,0], dtype=bool)))

    def test_connect_with_scalar_weights_and_delays(self):
        C = connectors.OneToOneConnector(weights=5.0, delays=0.5, safe=False)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(18, 80, 5.0, 0.5), (20, 82, 5, 0.5)])

    def test_connect_with_random_weights(self):
        rd = random.RandomDistribution(rng=MockRNG(num_processes=2, delta=1.0))
        C = connectors.OneToOneConnector(weights=rd, delays=0.5, safe=False)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(18, 80, 1.0, 0.5), (20, 82, 3.0, 0.5)])


class TestAllToAllConnector(object):

    def setup(self):
        self.prj = MockProjection(MockPre(4), MockPost(numpy.array([0,1,0,1,0], dtype=bool)))

    def test_connect_with_scalar_weights_and_delays(self):
        C = connectors.AllToAllConnector(weights=5.0, delays=0.5, safe=False)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        assert_equal(set(self.prj.connections),
                     set([(17, 80, 5.0, 0.5),
                          (17, 82, 5.0, 0.5),
                          (18, 80, 5.0, 0.5),
                          (18, 82, 5.0, 0.5),
                          (19, 80, 5.0, 0.5),
                          (19, 82, 5.0, 0.5),
                          (20, 80, 5.0, 0.5),
                          (20, 82, 5.0, 0.5)]))
    
    def test_connect_with_random_weights_parallel_safe(self):
        rd = random.RandomDistribution(rng=MockRNG(num_processes=2, delta=1.0))
        C = connectors.AllToAllConnector(weights=rd, delays=0.5, safe=False)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(17, 80, 1.0, 0.5),
                      (17, 82, 3.0, 0.5),
                      (18, 80, 6.0, 0.5),
                      (18, 82, 8.0, 0.5),
                      (19, 80, 11.0, 0.5),
                      (19, 82, 13.0, 0.5),
                      (20, 80, 16.0, 0.5),
                      (20, 82, 18.0, 0.5)])
        
    def test_connect_with_distance_dependent_weights_parallel_safe(self):
        d_expr = "d+100"
        C = connectors.AllToAllConnector(weights=d_expr, delays=0.5, safe=False)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(17, 80, 163.0, 0.5),   # 100+|17-80|
                      (17, 82, 165.0, 0.5),   # 100+|17-82|
                      (18, 80, 162.0, 0.5),   # etc.
                      (18, 82, 164.0, 0.5),
                      (19, 80, 161.0, 0.5),
                      (19, 82, 163.0, 0.5),
                      (20, 80, 160.0, 0.5),
                      (20, 82, 162.0, 0.5)])

    def test_connect_with_delays_None(self):
        C = connectors.AllToAllConnector(weights=0.1, delays=None)
        assert_equal(C.weights, 0.1)
        assert_equal(C.delays, None)
        assert C.safe
        assert C.allow_self_connections
        C.connect(self.prj)
        assert_equal(self.prj.connections[0][3], self.prj._simulator.state.min_delay)
        
    def test_connect_with_delays_too_small(self):
        C = connectors.AllToAllConnector(weights=0.1, delays=0.0)
        assert_raises(errors.ConnectionError, C.connect, self.prj)

    def test_connect_with_list_delays_too_small(self):
        C = connectors.AllToAllConnector(weights=0.1, delays=[1.0, 1.0, 0.0, 0.0, 3.0, 1.5, 2.3, 0.9])
        assert_raises(errors.ConnectionError, C.connect, self.prj)


class TestFixedProbabilityConnector(object):

    def setup(self):
        self.prj = MockProjection(MockPre(4),
                                  MockPost(numpy.array([0,1,0,1,0], dtype=bool)))

    def test_connect_with_default_args(self):
        C = connectors.FixedProbabilityConnector(p_connect=0.75)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        # 20 possible connections. Due to the mock RNG, only the
        # first 8 are created (17, 79), (17, 80), (17,81), (17,82), (17,83), (18,79), (18,80), (18,81)
        # of these, (17,80), (17,82), (18,80) are created on this node
        assert_equal(self.prj.connections,
                     [(17, 80, 0.0, MIN_DELAY),
                      (17, 82, 0.0, MIN_DELAY),
                      (18, 80, 0.0, MIN_DELAY)])

class TestDistanceDependentProbabilityConnector(object):

    def setup(self):
        self.prj = MockProjection(MockPre(4),
                                  MockPost(numpy.array([0,1,0,1,0], dtype=bool)))
        self.prj.rng = MockRNG(num_processes=2, delta=0.01)

    def test_connect_with_default_args(self):
        C = connectors.DistanceDependentProbabilityConnector(d_expression="d<62.5")
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        # 20 possible connections. Only those with a sufficiently small distance
        # are created
        assert_equal(self.prj.connections,
                     [(18, 80, 0.0, MIN_DELAY),
                      (19, 80, 0.0, MIN_DELAY),
                      (20, 80, 0.0, MIN_DELAY),
                      (20, 82, 0.0, MIN_DELAY)])

class TestFromListConnector(object):
    
    def setup(self):
        self.prj = MockProjection(MockPre(4),
                                  MockPost(numpy.array([0,1,0,1,0], dtype=bool)))
        
    def test_connect_with_valid_list(self):
        connection_list = [
            (0, 0, 0.1, 0.1),   # 17 -> 79
            (3, 0, 0.2, 0.11),  # 20 -> 79
            (2, 3, 0.3, 0.12),  # 19 -> 82  local
            (2, 2, 0.4, 0.13),  # 19 -> 81
            (0, 1, 0.5, 0.14),  # 17 -> 80 local
            ]
        C = connectors.FromListConnector(connection_list)
        C.progressbar = Mock()
        C.progression = Mock()
        C.connect(self.prj)
        # note that ListConnector does not filter out non-local connections
        assert_equal(self.prj.connections,
                     [(17, 79, 0.1, 0.1),
                      (17, 80, 0.5, 0.14),
                      (19, 82, 0.3, 0.12),
                      (19, 81, 0.4, 0.13),
                      (20, 79, 0.2, 0.11)])
        
    def test_connect_with_out_of_range_index(self):
        connection_list = [
            (0, 0, 0.1, 0.1),   # 17 -> 79
            (3, 0, 0.2, 0.11),  # 20 -> 79
            (2, 3, 0.3, 0.12),  # 19 -> 82  local
            (5, 2, 0.4, 0.13),  # NON-EXISTENT -> 81
            (0, 1, 0.5, 0.14),  # 17 -> 80 local
            ]
        C = connectors.FromListConnector(connection_list)
        assert_raises(errors.ConnectionError, C.connect, self.prj)


class TestFromFileConnector(object):
    
    def setup(self):
        self.prj = MockProjection(MockPre(4),
                                  MockPost(numpy.array([0,1,0,1,0], dtype=bool)))
        self.connection_list = [
            (0, 0, 0.1, 0.1),   # 17 -> 79
            (3, 0, 0.2, 0.11),  # 20 -> 79
            (2, 3, 0.3, 0.12),  # 19 -> 82  local
            (2, 2, 0.4, 0.13),  # 19 -> 81
            (0, 1, 0.5, 0.14),  # 17 -> 80 local
            ]
    
    def teardown(self):
        if os.path.exists("test.connections"):
            os.remove("test.connections")
    
    def test_connect_with_standard_text_file_not_distributed(self):
        numpy.savetxt("test.connections", self.connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=False)
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(17, 79, 0.1, 0.1),
                      (17, 80, 0.5, 0.14),
                      (19, 82, 0.3, 0.12),
                      (19, 81, 0.4, 0.13),
                      (20, 79, 0.2, 0.11)])
        
    def test_connect_with_standard_text_file_distributed(self):
        local_connection_list = [c for c in self.connection_list if c[1]%2 == 1]
        numpy.savetxt("test.connections.1", local_connection_list)
        C = connectors.FromFileConnector("test.connections", distributed=True)
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(17, 80, 0.5, 0.14),
                      (19, 82, 0.3, 0.12)])


class TestFixedNumberPostConnector(object):
    
    def setup(self):
        self.prj = MockProjection(MockPre(4),
                                  MockPost(numpy.array([0,1,0,1,0], dtype=bool)))
        self.prj.rng.rng = Mock()
        self.prj.rng.rng.permutation = lambda x: x
        
    def test_with_n_smaller_than_population_size(self):
        C = connectors.FixedNumberPostConnector(n=3)
        C.progressbar = Mock()
        C.progression = Mock()
        assert self.prj.rng is not None
        C.connect(self.prj)
        # FixedNumberPost does not currently filter out only local connections
        assert_equal(self.prj.connections,
                     [(17, 79, 0.0, MIN_DELAY),
                      (17, 80, 0.0, MIN_DELAY),
                      (17, 81, 0.0, MIN_DELAY),
                      (18, 79, 0.0, MIN_DELAY),
                      (18, 80, 0.0, MIN_DELAY),
                      (18, 81, 0.0, MIN_DELAY),
                      (19, 79, 0.0, MIN_DELAY),
                      (19, 80, 0.0, MIN_DELAY),
                      (19, 81, 0.0, MIN_DELAY),
                      (20, 79, 0.0, MIN_DELAY),
                      (20, 80, 0.0, MIN_DELAY),
                      (20, 81, 0.0, MIN_DELAY)])


class TestFixedNumberPreConnector(object):
    
    def setup(self):
        self.prj = MockProjection(MockPre(4),
                                  MockPost(numpy.array([0,1,0,1,0], dtype=bool)))
        self.prj.rng.rng = Mock()
        self.prj.rng.rng.permutation = lambda x: x
        
    def test_with_n_smaller_than_population_size(self):
        C = connectors.FixedNumberPreConnector(n=3)
        C.progressbar = Mock()
        C.progression = Mock()
        assert self.prj.rng is not None
        self.prj.post.local_cells = [MockCell(n) for n in self.prj.post.local_cells]
        C.connect(self.prj)
        assert_equal(self.prj.connections,
                     [(17, 80, 0.0, MIN_DELAY),
                      (18, 80, 0.0, MIN_DELAY),
                      (19, 80, 0.0, MIN_DELAY),
                      (17, 82, 0.0, MIN_DELAY),
                      (18, 82, 0.0, MIN_DELAY),
                      (19, 82, 0.0, MIN_DELAY)])
     

class TestDistanceMatrix(object):
    
    def test_really_simple0(self):
        A = numpy.zeros((3,))
        B = numpy.zeros((3,5))
        D = connectors.DistanceMatrix(B, space.Space())
        D.set_source(A)
        assert_arrays_equal(D.as_array(),
                            numpy.zeros((5,), float))

    def test_really_simple1(self):
        A = numpy.ones((3,))
        B = numpy.zeros((3,5))
        D = connectors.DistanceMatrix(B, space.Space())
        D.set_source(A)
        assert_arrays_equal(D.as_array(),
                            numpy.sqrt(3*numpy.ones((5,), float)))
