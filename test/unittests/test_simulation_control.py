from pyNN import common, errors
from nose.tools import assert_equal, assert_raises

class MockState(object):
    def __init__(self):
        self.accesses = []
    def __getattr__(self, name):
        if name == 'accesses':
            return self.__getattribute__(name)
        else:
            self.accesses.append(name)

class MockSimulator(object):
    def __init__(self):
        self.reset_called = False
        self.state = MockState()
    def reset(self):
        self.reset_called = True

class MockPopulation(common.Population):
    def __init__(self):
        self.initializations = []
    def initialize(self, variable, value):
        self.initializations.append((variable, value))

def test_setup():
    assert_raises(Exception, common.setup, min_delay=1.0, max_delay=0.9)
    assert_raises(Exception, common.setup, mindelay=1.0)  # } common
    assert_raises(Exception, common.setup, maxdelay=10.0) # } misspellings
    assert_raises(Exception, common.setup, dt=0.1)        # }
    assert_raises(Exception, common.setup, timestep=0.1, min_delay=0.09)
        
def test_end():
    assert_raises(NotImplementedError, common.end)
        
def test_run():
    assert_raises(NotImplementedError, common.run, 10.0)
               
def test_reset():
    common.simulator = MockSimulator()
    common.reset()
    assert common.simulator.reset_called
    
def test_initialize():
    p = MockPopulation()
    common.initialize(p, 'v', -65.0)
    assert p.initializations == [('v', -65.0)]
    
def test_current_time():
    common.simulator = MockSimulator()
    common.get_current_time()
    assert_equal(common.simulator.state.accesses, ['t'])
    
def test_time_step():
    common.simulator = MockSimulator()
    common.get_time_step()
    assert_equal(common.simulator.state.accesses, ['dt'])
    
def test_min_delay():
    common.simulator = MockSimulator()
    common.get_min_delay()
    assert_equal(common.simulator.state.accesses, ['min_delay'])

def test_max_delay():
    common.simulator = MockSimulator()
    common.get_max_delay()
    assert_equal(common.simulator.state.accesses, ['max_delay'])
    
def test_num_processes():
    common.simulator = MockSimulator()
    common.num_processes()
    assert_equal(common.simulator.state.accesses, ['num_processes'])
    
def test_rank():
    common.simulator = MockSimulator()
    common.rank()
    assert_equal(common.simulator.state.accesses, ['mpi_rank'])
