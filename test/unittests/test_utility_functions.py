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
    
def test_check_weight_with_scalar():
    assert_equal(4.3, common.check_weight(4.3, 'excitatory', is_conductance=True))
    assert_equal(4.3, common.check_weight(4.3, 'excitatory', is_conductance=False))
    assert_equal(4.3, common.check_weight(4.3, 'inhibitory', is_conductance=True))
    assert_equal(-4.3, common.check_weight(-4.3, 'inhibitory', is_conductance=False))
    assert_equal(common.DEFAULT_WEIGHT, common.check_weight(None, 'excitatory', is_conductance=True))
    assert_raises(errors.InvalidWeightError, common.check_weight, 4.3, 'inhibitory', is_conductance=False)
    assert_raises(errors.InvalidWeightError, common.check_weight, -4.3, 'inhibitory', is_conductance=True)
    assert_raises(errors.InvalidWeightError, common.check_weight, -4.3, 'excitatory', is_conductance=True)
    assert_raises(errors.InvalidWeightError, common.check_weight, -4.3, 'excitatory', is_conductance=False)
    
def test_check_weight_with_list():
    w = range(10)
    assert_equal(w, common.check_weight(w, 'excitatory', is_conductance=True).tolist())
    assert_equal(w, common.check_weight(w, 'excitatory', is_conductance=False).tolist())
    assert_equal(w, common.check_weight(w, 'inhibitory', is_conductance=True).tolist())
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'inhibitory', is_conductance=False)
    w = range(-10,0)
    assert_equal(w, common.check_weight(w, 'inhibitory', is_conductance=False).tolist())   
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'inhibitory', is_conductance=True)
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'excitatory', is_conductance=True)
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'excitatory', is_conductance=False)
    w = range(-5,5)
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'excitatory', is_conductance=True)
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'excitatory', is_conductance=False)
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'inhibitory', is_conductance=True)
    assert_raises(errors.InvalidWeightError, common.check_weight, w, 'inhibitory', is_conductance=False)

def test_check_weight_with_NaN():
    w = numpy.arange(10.0)
    w[0] = numpy.nan
    assert_arrays_equal(w[1:], common.check_weight(w, 'excitatory', is_conductance=True)[1:]) # NaN != NaN by definition
    
def test_check_weight_with_invalid_value():
    assert_raises(errors.InvalidWeightError, common.check_weight, "butterflies", 'excitatory', is_conductance=True)

def test_check_weight_is_conductance_is_None():
    # need to check that a log message was created
    assert_equal(4.3, common.check_weight(4.3, 'excitatory', is_conductance=None))
