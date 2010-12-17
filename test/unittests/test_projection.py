from pyNN import common, standardmodels
from nose.tools import assert_equal, assert_raises
from mock import Mock
import numpy
import os
from pyNN.utility import assert_arrays_equal

orig_rank = common.rank
orig_np = common.num_processes

def setup():
    common.rank = lambda: 1
    common.num_processes = lambda: 3

def teardown():
    common.rank = orig_rank
    common.num_processes = orig_np

class MockStandardCell(standardmodels.StandardCellType):
    recordable = ['v', 'spikes']

class MockPopulation(common.BasePopulation):
    label = "mock_population"
    first_id = 555

class MockConnectionManager(object):
    def __len__(self):
        return 999
    def __getitem__(self, i):
        return 888+i
    def get(self, name, format):
        return numpy.arange(100)

class MockConnection(object):
    source = 246
    target = 652
    weight = 542
    delay = 254

def test_create_simple():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    
def test_create_with_synapse_dynamics():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock(),
                            synapse_dynamics=standardmodels.SynapseDynamics())
    
def test_len():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    assert_equal(len(prj), len(prj.connection_manager))
    
def test_size_no_gather():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    assert_equal(prj.size(gather=False), len(prj))
    
def test_size_with_gather():
    orig_mpi_sum = common.recording.mpi_sum
    common.recording.mpi_sum = Mock()
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.size(gather=True)
    common.recording.mpi_sum.assert_called_with(len(prj))
    common.recording.mpi_sum = orig_mpi_sum
    
def test__getitem():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    assert_equal(prj[0], 888)
    
def test_set_weights():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.synapse_type = "foo"
    prj.post.local_cells = [0]
    prj.connection_manager = MockConnectionManager()
    prj.connection_manager.set = Mock()
    prj.setWeights(0.5)
    prj.connection_manager.set.assert_called_with('weight', 0.5)
    
def test_randomize_weights():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.setWeights = Mock()
    rd = Mock()
    rd.next = Mock(return_value=777)
    prj.randomizeWeights(rd)
    rd.next.assert_called_with(len(prj))
    prj.setWeights.assert_called_with(777)
    
def test_set_delays():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.connection_manager.set = Mock()
    prj.setDelays(0.5)
    prj.connection_manager.set.assert_called_with('delay', 0.5)
    
def test_randomize_delays():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.setDelays = Mock()
    rd = Mock()
    rd.next = Mock(return_value=777)
    prj.randomizeDelays(rd)
    rd.next.assert_called_with(len(prj))
    prj.setDelays.assert_called_with(777)
    
def test_set_synapse_dynamics_param():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.connection_manager.set = Mock()
    prj.setSynapseDynamics('U', 0.5)
    prj.connection_manager.set.assert_called_with('U', 0.5)
    
def test_get_weights():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.connection_manager.get = Mock()
    prj.getWeights(format='list', gather=False)
    prj.connection_manager.get.assert_called_with('weight', 'list')
    
def test_get_delays():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.connection_manager.get = Mock()
    prj.getDelays(format='list', gather=False)
    prj.connection_manager.get.assert_called_with('delay', 'list')

def test_save_connections():
    filename = "test.connections"
    if os.path.exists(filename + ".1"):
        os.remove(filename + ".1")
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.connections = [MockConnection(), MockConnection(), MockConnection()]
    prj.saveConnections(filename, gather=False, compatible_output=False)
    assert os.path.exists(filename + ".1")
    os.remove(filename + ".1")

def test_print_weights_as_list():
    filename = "test.weights"
    if os.path.exists(filename):
        os.remove(filename)
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.printWeights(filename, format='list', gather=False)
    assert os.path.exists(filename)
    os.remove(filename)
    
def test_print_weights_as_array():
    filename = "test.weights"
    if os.path.exists(filename):
        os.remove(filename)
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.connection_manager = MockConnectionManager()
    prj.printWeights(filename, format='array', gather=False)
    assert os.path.exists(filename)
    os.remove(filename)

def test_weight_histogram_with_args():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.getWeights = Mock(return_value=numpy.array(range(10)*42))
    n, bins = prj.weightHistogram(min=0.0, max=9.0, nbins=10)
    assert_equal(n.size, 10)
    assert_equal(bins.size, n.size+1)
    assert_arrays_equal(n, 42*numpy.ones(10))
    assert_equal(n.sum(), 420)
    assert_arrays_equal(bins, numpy.arange(0.0, 9.1, 0.9))

def test_weight_histogram_no_args():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    prj.getWeights = Mock(return_value=numpy.array(range(10)*42))
    n, bins = prj.weightHistogram(nbins=10)
    assert_equal(n.size, 10)
    assert_equal(bins.size, n.size+1)
    assert_arrays_equal(n, 42*numpy.ones(10))
    assert_equal(n.sum(), 420)
    assert_arrays_equal(bins, numpy.arange(0.0, 9.1, 0.9))

def test_describe():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock(), synapse_dynamics=standardmodels.SynapseDynamics())
    prj.connection_manager = MockConnectionManager()
    prj.pre.describe = Mock()
    prj.post.describe = Mock()
    assert isinstance(prj.describe(engine='string'), basestring)
    assert isinstance(prj.describe(template=None), dict)
