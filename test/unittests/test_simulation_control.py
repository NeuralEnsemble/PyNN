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
    assert_raises(NotImplementedError, common.control.end)
        
def test_run():
    assert_raises(NotImplementedError, common.control.run, 10.0)
               
def test_reset():
    simulator = MockSimulator()
    reset = common.build_reset(simulator)
    reset()
    assert simulator.reset_called
    
def test_initialize():
    p = MockPopulation()
    common.initialize(p, 'v', -65.0)
    assert p.initializations == [('v', -65.0)]


def test_current_time():
    simulator = MockSimulator()
    get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank = common.build_state_queries(simulator)
    get_current_time()
    assert_equal(simulator.state.accesses, ['t'])
    
def test_time_step():
    simulator = MockSimulator()
    get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank = common.build_state_queries(simulator)
    get_time_step()
    assert_equal(simulator.state.accesses, ['dt'])
    
def test_min_delay():
    simulator = MockSimulator()
    get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank = common.build_state_queries(simulator)
    get_min_delay()
    assert_equal(simulator.state.accesses, ['min_delay'])

def test_max_delay():
    simulator = MockSimulator()
    get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank = common.build_state_queries(simulator)
    get_max_delay()
    assert_equal(simulator.state.accesses, ['max_delay'])
    
def test_num_processes():
    simulator = MockSimulator()
    get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank = common.build_state_queries(simulator)
    num_processes()
    assert_equal(simulator.state.accesses, ['num_processes'])
    
def test_rank():
    simulator = MockSimulator()
    get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank = common.build_state_queries(simulator)
    rank()
    assert_equal(simulator.state.accesses, ['mpi_rank'])
