from pyNN import common, standardmodels
from nose.tools import assert_equal, assert_raises
from mock import Mock

common.rank = lambda: 1
common.num_processes = lambda: 3

class MockStandardCell(standardmodels.StandardCellType):
    recordable = ['v', 'spikes']

class MockPopulation(common.BasePopulation):
    #size = 13
    #all_cells = numpy.arange(13)
    #_mask_local = numpy.array([0,1,0,1,0,1,0,1,0,1,0,1,0], bool)
    #local_cells = all_cells[_mask_local]
    #positions = numpy.arange(39).reshape((13,3)).T
    label = "mock_population"
    #celltype = MockStandardCell({})

def test_create_simple():
    p1 = MockPopulation()
    p2 = MockPopulation()
    prj = common.Projection(p1, p2, method=Mock())
    
#def test_create_with_synapse_dynamics():
    