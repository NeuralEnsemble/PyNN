from pyNN import common, standardmodels
from nose.tools import assert_equal, assert_raises
import numpy
from mock import Mock, patch
from pyNN.utility import assert_arrays_equal

class MockID(object):
    def __init__(self, i, parent):
        self.label = str(i)
        self.parent = parent
    def get_parameters(self):
        return {}
    
class MockPopulation(common.Population):
    recorder_class = Mock()
    initialize = Mock()
    
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


# test create with population parent and mask selector
def test_create_with_slice_selector():
    p = MockPopulation(11, MockStandardCell)
    mask = slice(3,9,2)
    pv = common.PopulationView(parent=p, selector=mask)
    assert_equal(pv.parent, p)
    assert_equal(pv.size, 3)
    assert_equal(pv.mask, mask)
    assert_arrays_equal(pv.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
    assert_arrays_equal(pv.local_cells, numpy.array([p.all_cells[3]]))
    assert_arrays_equal(pv._mask_local, numpy.array([1,0,0], dtype=bool))
    assert_equal(pv.celltype, p.celltype)
    assert_equal(pv.cellparams, p.cellparams)
    assert_equal(pv.recorders, p.recorders)
    assert_equal(pv.first_id, p.all_cells[3])
    assert_equal(pv.last_id, p.all_cells[7])

def test_create_with_boolean_array_selector():
    p = MockPopulation(11, MockStandardCell)
    mask = numpy.array([0,0,0,1,0,1,0,1,0,0,0], dtype=bool)
    pv = common.PopulationView(parent=p, selector=mask)
    assert_arrays_equal(pv.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
    #assert_arrays_equal(pv.mask, mask)

def test_create_with_index_array_selector():
    p = MockPopulation(11, MockStandardCell)
    mask = numpy.array([3, 5, 7])
    pv = common.PopulationView(parent=p, selector=mask)
    assert_arrays_equal(pv.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
    assert_arrays_equal(pv.mask, mask)

# test create with populationview parent and mask selector
def test_create_with_slice_selector():
    p = MockPopulation(11, MockStandardCell)
    mask1 = slice(0,9,1)
    pv1 = common.PopulationView(parent=p, selector=mask1)
    assert_arrays_equal(pv1.all_cells, p.all_cells[0:9])
    mask2 = slice(3,9,2)
    pv2 = common.PopulationView(parent=pv1, selector=mask2)
    assert_equal(pv2.parent, pv1) # or would it be better to resolve the parent chain up to an actual Population?
    assert_arrays_equal(pv2.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
    assert_arrays_equal(pv2._mask_local, numpy.array([1,0,0], dtype=bool))

# test initial values property

def test_structure_property():
    p = MockPopulation(11, MockStandardCell)
    mask = slice(3,9,2)
    pv = common.PopulationView(parent=p, selector=mask)
    assert_equal(pv.structure, p.structure)

# test positions property
def test_get_positions():
    p = MockPopulation(11, MockStandardCell)
    ppos = numpy.random.uniform(size=(3,11))
    p._positions = ppos
    pv = common.PopulationView(parent=p,
                               selector=slice(3,9,2))
    assert_arrays_equal(pv.positions, numpy.array([ppos[:,3], ppos[:,5], ppos[:,7]]).T)

# test id_to_index
def test_id_to_index():
    p = MockPopulation(11, MockStandardCell)
    mask = slice(3,9,2)
    pv = common.PopulationView(parent=p, selector=mask)
    assert_equal(pv.id_to_index(p.all_cells[3]), 0)
    assert_equal(pv.id_to_index(p.all_cells[7]), 2)
    assert_raises(IndexError, pv.id_to_index, p.all_cells[0])
    
# test describe
def test_describe():
    p = MockPopulation(11, MockStandardCell)
    mask = slice(3,9,2)
    pv = common.PopulationView(parent=p, selector=mask)
    assert isinstance(pv.describe(), basestring)
    assert isinstance(pv.describe(template=None), dict)
