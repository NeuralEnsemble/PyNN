from pyNN import common, errors
import numpy
from nose.tools import assert_equal, assert_raises
from pyNN.utility import assert_arrays_equal

MIN_DELAY = 1.23
MAX_DELAY = 999


class MockCell(object):
    def __init__(self, cellclass, local=True):
        self.celltype = cellclass()
        self.local = local

def build_cellclass(cb):
    class MockCellClass(object):
        conductance_based = cb
    return MockCellClass

class MockSimulator(object):
    class MockState(object):
        min_delay = MIN_DELAY
        max_delay = MAX_DELAY
    state = MockState()
    
def setup():
    common.control.simulator = MockSimulator

def test_is_conductance():
    for cb in (True, False):
        cell = MockCell(build_cellclass(cb))
        assert common.is_conductance(cell) == cb
    
def test_is_conductance_with_nonlocal_cell():
    cell = MockCell(build_cellclass(True), local=False)
    assert common.is_conductance(cell) is None
    

