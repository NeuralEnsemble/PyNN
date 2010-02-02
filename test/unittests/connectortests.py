from mpi4py import MPI
import numpy
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.connectors2 import FixedProbabilityConnector, AllToAllConnector
from pyNN import common

mpi_comm = MPI.COMM_WORLD

class MockSimulatorModule(object):
    class State(object):
        min_delay = 0.1
    state = State()

common.simulator = MockSimulatorModule()

class MockProjection(object):
    
    class MockConnectionManager(object):
        def connect(self, src, targets, weights, delays):
            assert len(targets) == len(weights) == len(delays), "%d %d %d" % (len(targets), len(weights), len(delays))
    connection_manager = MockConnectionManager()
    
    def __init__(self, pre, post, rng):
        self.pre = pre
        self.post = post
        self.rng = rng

class MockID(int):
    
    @property
    def position(self):
        return numpy.array([float(self), 0.0, 0.0])
        

class MockPopulation(object):
    
    def __init__(self, n):
        self.all_cells = numpy.array([MockID(i) for i in range(n)], dtype=MockID)
        self.positions = numpy.array([(i, 0.0, 0.0) for i in self.all_cells], dtype=float).T
        self._mask_local = numpy.array([i%mpi_comm.size == mpi_comm.rank for i in range(n)])
        self.local_cells = self.all_cells[self._mask_local]
    
    def all(self):
        return self.all_cells
    
    @property
    def size(self):
        return self.all_cells.size
    
p1 = MockPopulation(100)
p2 = MockPopulation(100)

rng = NumpyRNG(8569552)

weight_sources = [0.1,
                  RandomDistribution('uniform', (0,1), rng),
                  numpy.arange(0.0, 1.0, 1e-4).reshape(100,100),
                  "exp(-(d*d)/1e4)"]

for weight_source in weight_sources:
    connector = FixedProbabilityConnector(p_connect=0.3,
                                          allow_self_connections=True,
                                          weights=weight_source,
                                          delays=0.2)
    prj = MockProjection(p1, p2, rng)
    connector.connect(prj)
    connector = AllToAllConnector(allow_self_connections=True,
                                  weights=weight_source,
                                  delays=0.2)
    prj = MockProjection(p1, p2, rng)
    connector.connect(prj)
