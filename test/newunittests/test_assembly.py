
from pyNN.common import Assembly, BasePopulation
from nose.tools import assert_equal, assert_raises
import numpy
from tools import assert_arrays_equal
from mock import Mock
    

def test_create_with_zero_populations():
    a = Assembly()
    assert_equal(a.populations, [])
    assert isinstance(a.label, basestring)

class MockPopulation(BasePopulation):
    size = 10
    local_cells = range(1,10,2)
    all_cells = range(10)
    initialize = Mock()
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
    assert_equal(a3.populations, [p1, p2, p2, p3]) # or do we want [p1, p2, p3]?

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
    assert_equal(a1.populations, [p1, p2, p2, p3])
    
def test_initialize():
    p1 = MockPopulation()
    p2 = MockPopulation()
    a = Assembly(p1, p2)
    a.initialize("v", -54.3)
    p1.initialize.assert_called_with("v", -54.3)
    p2.initialize.assert_called_with("v", -54.3)

#test record

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
