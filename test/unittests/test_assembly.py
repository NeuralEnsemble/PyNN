from pyNN import common
from pyNN.common.populations import Assembly, BasePopulation
from nose.tools import assert_equal, assert_raises
import numpy
from pyNN.utility import assert_arrays_equal
from mock import Mock
    

class MockSimulator(object):
    class MockState(object):
        mpi_rank = 1
        num_processes = 2
    state = MockState()

def test_create_with_zero_populations():
    a = Assembly()
    assert_equal(a.populations, [])
    assert isinstance(a.label, basestring)

class MockPopulation(BasePopulation):
    _simulator = MockSimulator
    size = 10
    local_cells = numpy.arange(_simulator.state.mpi_rank,10,_simulator.state.num_processes)
    all_cells = numpy.arange(size)
    _mask_local = numpy.arange(size)%_simulator.state.num_processes == _simulator.state.mpi_rank
    initialize = Mock()
    positions = numpy.arange(3*size).reshape(3,size)
    def describe(self, template='abcd', engine=None):
        if template is None:
            return {'label': 'dummy'}
        else:
            return ""

def test_create_with_one_population():
    p = MockPopulation()
    a = Assembly(p)
    assert_equal(a.populations, [p])
    assert isinstance(a.label, basestring)

def test_create_with_two_populations():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assert_equal(a.populations, [p1, p2])
    assert_equal(a.label, "test")

def test_create_with_non_population_should_raise_Exception():
    assert_raises(TypeError, Assembly, [1, 2, 3])

def test_size_property():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assert_equal(a.size, p1.size + p2.size)

def test_positions_property():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assert_arrays_equal(a.positions, numpy.concatenate((p1.positions, p2.positions), axis=1))

def test__len__():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assert_equal(len(a), len(p1) + len(p2))

def test_local_cells():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assert_arrays_equal(a.local_cells, numpy.append(p1.local_cells, p2.local_cells))

def test_all_cells():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assert_arrays_equal(a.all_cells, numpy.append(p1.all_cells, p2.all_cells))

def test_iter():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2, label="test")
    assembly_ids = [id for id in a]

def test__add__population():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a1 = Assembly(p1)
    assert_equal(a1.populations, [p1])
    a2 = a1 + p2
    assert_equal(a1.populations, [p1])
    assert_equal(a2.populations, [p1, p2])

def test__add__assembly():
    p1 = MockPopulation()
    p2 = MockPopulation()
    p3 = MockPopulation()
    a1 = Assembly(p1, p2)
    a2 = Assembly(p2, p3)
    a3 = a1 + a2
    assert_equal(a3.populations, [p1, p2, p3]) # or do we want [p1, p2, p3]?

def test_add_inplace_population():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1)
    a += p2
    assert_equal(a.populations, [p1, p2])
    
def test_add_inplace_assembly():
    p1 = MockPopulation()
    p2 = MockPopulation()
    p3 = MockPopulation()
    a1 = Assembly(p1, p2)
    a2 = Assembly(p2, p3)
    a1 += a2
    assert_equal(a1.populations, [p1, p2, p3])

def test_add_invalid_object():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2)
    assert_raises(TypeError, a.__add__, 42)
    assert_raises(TypeError, a.__iadd__, 42)

def test_initialize():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2)
    a.initialize("v", -54.3)
    p1.initialize.assert_called_with("v", -54.3)
    p2.initialize.assert_called_with("v", -54.3)
    
def test_describe():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2)
    assert isinstance(a.describe(), basestring)
    assert isinstance(a.describe(template=None), dict)

def test_get_population():
    p1 = MockPopulation()
    p1.label = "pop1"
    p2 = MockPopulation()
    p2.label = "pop2"
    a = Assembly(p1, p2)
    assert_equal(a.get_population("pop1"), p1)
    assert_equal(a.get_population("pop2"), p2)
    assert_raises(KeyError, a.get_population, "foo")

def test_all_cells():
    p1 = MockPopulation()
    p2 = MockPopulation()
    p3 = MockPopulation()
    a = Assembly(p1, p2, p3)
    assert_equal(a.all_cells.size,
                 p1.all_cells.size + p2.all_cells.size + p3.all_cells.size)
    assert_equal(a.all_cells[0], p1.all_cells[0])
    assert_equal(a.all_cells[-1], p3.all_cells[-1])
    assert_arrays_equal(a.all_cells, numpy.append(p1.all_cells, (p2.all_cells, p3.all_cells)))
    
def test_local_cells():
    p1 = MockPopulation()
    p2 = MockPopulation()
    p3 = MockPopulation()
    a = Assembly(p1, p2, p3)
    assert_equal(a.local_cells.size,
                 p1.local_cells.size + p2.local_cells.size + p3.local_cells.size)
    assert_equal(a.local_cells[0], p1.local_cells[0])
    assert_equal(a.local_cells[-1], p3.local_cells[-1])
    assert_arrays_equal(a.local_cells, numpy.append(p1.local_cells, (p2.local_cells, p3.local_cells)))

def test_mask_local():
    p1 = MockPopulation()
    p2 = MockPopulation()
    p3 = MockPopulation()
    a = Assembly(p1, p2, p3)
    assert_equal(a._mask_local.size,
                 p1._mask_local.size + p2._mask_local.size + p3._mask_local.size)
    assert_equal(a._mask_local[0], p1._mask_local[0])
    assert_equal(a._mask_local[-1], p3._mask_local[-1])
    assert_arrays_equal(a._mask_local, numpy.append(p1._mask_local, (p2._mask_local, p3._mask_local)))
    assert_arrays_equal(a.local_cells, a.all_cells[a._mask_local])

def test_save_positions():
    import os
    Assembly._simulator = MockSimulator
    Assembly._simulator.state.mpi_rank = 0
    p1 = MockPopulation()
    p2 = MockPopulation()
    p1.all_cells = numpy.array([34, 45])
    p2.all_cells = numpy.array([56, 67])
    p1.positions = numpy.arange(0,6).reshape((2,3)).T
    p2.positions = numpy.arange(6,12).reshape((2,3)).T
    a = Assembly(p1, p2, label="test")
    output_file = Mock()
    a.save_positions(output_file)
    assert_arrays_equal(output_file.write.call_args[0][0],
                        numpy.array([[34, 0, 1, 2], [45, 3, 4, 5], [56, 6, 7, 8], [67, 9, 10, 11]]))
    assert_equal(output_file.write.call_args[0][1], {'assembly': a.label})
    # arguably, the first column should contain indices, not ids.
    del Assembly._simulator