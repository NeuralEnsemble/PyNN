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
    
    def _create_cells(self, cellclass, cellparams, size):
        self.all_cells = numpy.array([MockID(i, self) for i in range(size)], MockID)
        self._mask_local = numpy.arange(size)%5==3

class MockStandardCell(standardmodels.StandardCellType):
    default_parameters = {
        'a': 20.0,
        'b': -34.9
    }
    translations = standardmodels.build_translations(('a', 'A'), ('b', 'B'))


def test_create_population_standard_cell_simple():
    p = MockPopulation(11, MockStandardCell)
    assert_equal(p.size, 11)
    assert_equal(p.label, 'population0')
    assert isinstance(p.celltype, MockStandardCell)
    assert isinstance(p._structure, space.Line)
    assert_equal(p._positions, None)
    assert_equal(p.cellparams, None) #? shouldn't we fill in the default values?
    assert_equal(p.initial_values, {})
    assert isinstance(p.recorders, dict)
    
def test_create_population_standard_cell_with_params():
    p = MockPopulation(11, MockStandardCell, {'a': 17.0, 'b': 0.987})
    assert isinstance(p.celltype, MockStandardCell)
    assert_equal(p.cellparams, {'a': 17.0, 'b': 0.987})

# test create native cell

# test create native cell with params

# test create with structure

# test local_cells property

# test id_to_index

# test id_to_local_index

# test structure property

# test positions property

# test describe method

