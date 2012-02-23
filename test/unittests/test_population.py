from pyNN import errors, random, standardmodels, space
from pyNN.common import populations
from nose.tools import assert_equal, assert_raises
import numpy
from mock import Mock, patch
from pyNN.utility import assert_arrays_equal


class MockSimulator(object):
    class MockState(object):
        mpi_rank = 1
        num_processes = 3
    state = MockState()

class MockID(int, populations.IDMixin):
    def __init__(self, n):
        int.__init__(n)
        populations.IDMixin.__init__(self)
    def get_parameters(self):
        return {}

class MockPopulation(populations.Population):
    _simulator = MockSimulator
    recorder_class = Mock()
    initialize = Mock()
    
    def _get_view(self, selector, label=None):
        return populations.PopulationView(self, selector, label)
    
    def _create_cells(self, cellclass, cellparams, size):
        self.all_cells = numpy.array([MockID(i) for i in range(999, 999+size)], MockID)
        self.cell      = self.all_cells
        self._mask_local = numpy.arange(size)%5==3 # every 5th cell, starting with the 4th, is on this node
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]

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
    assert_equal(p.celltype.parameters, {'A': 20.0, 'B': -34.9})
    assert_equal(p.initial_values, {})
    assert isinstance(p.recorders, dict)
    p.initialize.assert_called_with('m', -1.23)
    
def test_create_population_standard_cell_with_params():
    p = MockPopulation(11, MockStandardCell, {'a': 17.0, 'b': 0.987})
    assert isinstance(p.celltype, MockStandardCell)
    assert_equal(p.celltype.parameters, {'A': 17.0, 'B': 0.987})

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

def test_create_with_initial_values():
    p = MockPopulation(11, MockStandardCell, initial_values={'m': -67.8})
    p.initialize.assert_called_with('m', -67.8)

# test local_cells property

def test_cell_property():
    p = MockPopulation(11, MockStandardCell)
    assert_arrays_equal(p.cell, p.all_cells)

def test_id_to_index():
    p = MockPopulation(11, MockStandardCell)
    assert isinstance(p[0], populations.IDMixin)
    assert_equal(p.id_to_index(p[0]), 0)
    assert_equal(p.id_to_index(p[10]), 10)

def test_id_to_index_with_array():
    p = MockPopulation(11, MockStandardCell)
    assert isinstance(p[0], populations.IDMixin)
    assert_arrays_equal(p.id_to_index(p.all_cells[3:9:2]), numpy.arange(3,9,2))

def test_id_to_index_with_populationview():
    p = MockPopulation(11, MockStandardCell)
    assert isinstance(p[0], populations.IDMixin)
    view = p[3:7]
    assert isinstance(view, populations.PopulationView)
    assert_arrays_equal(p.id_to_index(view), numpy.arange(3,7))

def test_id_to_index_with_invalid_id():
    p = MockPopulation(11, MockStandardCell)
    assert isinstance(p[0], populations.IDMixin)
    assert_raises(ValueError, p.id_to_index, MockID(p.last_id+1))
    assert_raises(ValueError, p.id_to_index, MockID(p.first_id-1))
    
def test_id_to_index_with_invalid_ids():
    p = MockPopulation(11, MockStandardCell)
    assert_raises(ValueError, p.id_to_index, [MockID(p.first_id-1)] + p.all_cells[0:3].tolist())

def test_id_to_local_index():
    orig_np = MockPopulation._simulator.state.num_processes
    MockPopulation._simulator.state.num_processes = 5
    p = MockPopulation(11, MockStandardCell)
    # every 5th cell, starting with the 4th, is on this node.
    assert_equal(p.id_to_local_index(p[3]), 0)
    assert_equal(p.id_to_local_index(p[8]), 1)
    
    MockPopulation._simulator.state.num_processes = 1
    # only one node
    assert_equal(p.id_to_local_index(p[3]), 3)
    assert_equal(p.id_to_local_index(p[8]), 8)
    MockPopulation._simulator.state.num_processes = orig_np

def test_id_to_local_index_with_invalid_id():
    orig_np = MockPopulation._simulator.state.num_processes
    MockPopulation._simulator.state.num_processes = 5
    p = MockPopulation(11, MockStandardCell)
    # every 5th cell, starting with the 4th, is on this node.
    assert_raises(ValueError, p.id_to_local_index, p[0])
    MockPopulation._simulator.state.num_processes = orig_np

# test structure property
def test_set_structure():
    p = MockPopulation(11, MockStandardCell)
    p._positions = numpy.arange(33).reshape(3,11)
    new_struct = MockStructure()
    p.structure = new_struct
    assert_equal(p._structure, new_struct)
    assert_equal(p._positions, None)

# test positions property
def test_get_positions():
    p = MockPopulation(11, MockStandardCell)
    pos1 = numpy.arange(33).reshape(3,11)
    p._structure = Mock()
    p._structure.generate_positions = Mock(return_value=pos1)
    assert_equal(p._positions, None)
    assert_arrays_equal(p.positions, pos1)
    assert_arrays_equal(p._positions, pos1)
    
    pos2 = 1+numpy.arange(33).reshape(3,11)
    p._positions = pos2
    assert_arrays_equal(p.positions, pos2)

def test_set_positions():
    p = MockPopulation(11, MockStandardCell)
    assert p._structure != None
    new_positions = numpy.random.uniform(size=(3,11))
    p.positions = new_positions
    assert_equal(p.structure, None)
    assert_arrays_equal(p.positions, new_positions)
    new_positions[0,0] = 99.9
    assert p.positions[0,0] != 99.9

def test_position_generator():
    p = MockPopulation(11, MockStandardCell)
    assert_arrays_equal(p.position_generator(0), p.positions[:,0])
    assert_arrays_equal(p.position_generator(10), p.positions[:,10])
    assert_arrays_equal(p.position_generator(-1), p.positions[:,10])
    assert_arrays_equal(p.position_generator(-11), p.positions[:,0])
    assert_raises(IndexError, p.position_generator, 11)
    assert_raises(IndexError, p.position_generator, -12)

# test describe method
def test_describe():
    p = MockPopulation(11, MockStandardCell)
    assert isinstance(p.describe(), basestring)
    assert isinstance(p.describe(template=None), dict)
