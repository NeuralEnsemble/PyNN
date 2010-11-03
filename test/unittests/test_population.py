from pyNN import common, errors, random, standardmodels, space
from nose.tools import assert_equal, assert_raises
import numpy
from mock import Mock, patch
from tools import assert_arrays_equal


class MockID(object):
    def __init__(self, i, parent):
        self.label = str(i)
        self.parent = parent
    def get_parameters(self):
        return {}

class MockPopulation(common.Population):
    recorder_class = Mock()
    initialize = Mock()
    first_id = 999
    last_id = 7777
    
    def _create_cells(self, cellclass, cellparams, size):
        self.all_cells = numpy.array([MockID(i, self) for i in range(size)], MockID)
        self._mask_local = numpy.arange(size)%5==3

class MockStandardCell(standardmodels.StandardCellType):
    default_parameters = {
        'a': 20.0,
        'b': -34.9
    }
    translations = standardmodels.build_translations(('a', 'A'), ('b', 'B'))
    default_initial_values = {'m': -1.23}

class MockStructure(space.BaseStructure):
    parameter_names = ('p0', 'p1')
    p0 = 1
    p1 = 2


def test_create_population_standard_cell_simple():
    p = MockPopulation(11, MockStandardCell)
    assert_equal(p.size, 11)
    assert isinstance(p.label, basestring)
    assert isinstance(p.celltype, MockStandardCell)
    assert isinstance(p._structure, space.Line)
    assert_equal(p._positions, None)
    assert_equal(p.cellparams, None) #? shouldn't we fill in the default values?
    assert_equal(p.initial_values, {})
    assert isinstance(p.recorders, dict)
    p.initialize.assert_called_with('m', -1.23)
    
def test_create_population_standard_cell_with_params():
    p = MockPopulation(11, MockStandardCell, {'a': 17.0, 'b': 0.987})
    assert isinstance(p.celltype, MockStandardCell)
    assert_equal(p.cellparams, {'a': 17.0, 'b': 0.987})

# test create native cell

# test create native cell with params

# test create with structure
def test_create_population_with_implicit_grid():
    p = MockPopulation((11,), MockStandardCell)
    assert_equal(p.size, 11)
    assert isinstance(p.structure, space.Line)
    p = MockPopulation((5,6), MockStandardCell)
    assert_equal(p.size, 30)
    assert isinstance(p.structure, space.Grid2D)
    p = MockPopulation((2,3,4), MockStandardCell)
    assert_equal(p.size, 24)
    assert isinstance(p.structure, space.Grid3D)
    assert_raises(Exception, MockPopulation, (2,3,4,5), MockStandardCell)

# test local_cells property

def test_cell_property():
    p = MockPopulation(11, MockStandardCell)
    assert_arrays_equal(p.cell, p.all_cells)

#def test_id_to_index():

# test id_to_local_index

# test structure property
def test_set_structure():
    p = MockPopulation(11, MockStandardCell)
    p._positions = "some positions"
    new_struct = MockStructure()
    p.structure = new_struct
    assert_equal(p._structure, new_struct)
    assert_equal(p._positions, None)

# test positions property
def test_get_positions():
    p = MockPopulation(11, MockStandardCell)
    p._structure = Mock()
    p._structure.generate_positions = Mock(return_value="some positions")
    assert_equal(p._positions, None)
    assert_equal(p.positions, "some positions")
    assert_equal(p._positions, "some positions")
    
    p._positions = "some other positions"
    assert_equal(p.positions, "some other positions")

def test_set_positions():
    p = MockPopulation(11, MockStandardCell)
    assert p._structure != None
    new_positions = numpy.random.uniform(size=(3,11))
    p.positions = new_positions
    assert_equal(p.structure, None)
    assert_arrays_equal(p.positions, new_positions)
    new_positions[0,0] = 99.9
    assert p.positions[0,0] != 99.9

# test describe method
def test_describe():
    p = MockPopulation(11, MockStandardCell)
    assert isinstance(p.describe(), basestring)
    assert isinstance(p.describe(template=None), dict)
