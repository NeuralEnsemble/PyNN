from pyNN import common, errors, random, standardmodels, recording
from pyNN.common import populations
from nose.tools import assert_equal, assert_raises
import numpy
from mock import Mock, patch
from pyNN.utility import assert_arrays_equal
from pyNN import core
    
builtin_open = open
id_map = {'larry': 0, 'curly': 1, 'moe': 2, 'joe': 3, 'william': 4, 'jack': 5, 'averell': 6}


class MockSimulator(object):
    class MockState(object):
        mpi_rank = 1
        num_processes = 3
    state = MockState()

class MockStandardCell(standardmodels.StandardCellType):
    recordable = ['v', 'spikes']
    default_parameters = {'tau_m': 999.9, 'i_offset': 321.0, 'spike_times': [0,1,2], 'foo': 33.3}
    translations = {'tau_m': None, 'i_offset': None, 'spike_times': None, 'foo': None}
    @classmethod
    def translate(cls, parameters):
        return parameters

class MockPopulation(populations.BasePopulation):
    _simulator = MockSimulator
    size = 13
    all_cells = numpy.arange(100, 113)
    _mask_local = numpy.array([0,1,0,1,0,1,0,1,0,1,0,1,0], bool)
    local_cells = all_cells[_mask_local]
    positions = numpy.arange(39).reshape((13,3)).T
    label = "mock_population"
    celltype = MockStandardCell({})
    initial_values = {"foo": core.LazyArray(numpy.array((98, 100, 102)), shape=(3,))}
    assembly_class = populations.Assembly

    def _get_view(self, selector, label=None):
        return populations.PopulationView(self, selector, label)

    def id_to_index(self, id):
        if id.label in id_map:
            return id_map[id.label]
        else:
            raise Exception("Invalid ID")
        
    def id_to_local_index(self, id):
        if id.label in id_map:
            global_index = id_map[id.label]
            if global_index%2 == 1:
                return global_index/2
            else:
                raise Exception("ID not on this node")
        else:
            raise Exception("Invalid ID")

class MockID(object):
    def __init__(self, label, parent):
        self.label = label
        self.parent = parent

def test__getitem__int():
    p = MockPopulation()
    assert_equal(p[0], 100)
    assert_equal(p[12], 112)
    assert_raises(IndexError, p.__getitem__, 13)
    assert_equal(p[-1], 112)
    
def test__getitem__slice():
    orig_PV = populations.PopulationView
    populations.PopulationView = Mock()
    p = MockPopulation()
    pv = p[3:9]
    populations.PopulationView.assert_called_with(p, slice(3,9,None), None)
    populations.PopulationView = orig_PV

def test__getitem__list():
    orig_PV = populations.PopulationView
    populations.PopulationView = Mock()
    p = MockPopulation()
    pv = p[range(3,9)]
    populations.PopulationView.assert_called_with(p, range(3,9), None)
    populations.PopulationView = orig_PV

def test__getitem__tuple():
    orig_PV = populations.PopulationView
    populations.PopulationView = Mock()
    p = MockPopulation()
    pv = p[(3,5,7)]
    populations.PopulationView.assert_called_with(p, [3,5,7], None)
    populations.PopulationView = orig_PV

def test__getitem__invalid():
    p = MockPopulation()
    assert_raises(TypeError, p.__getitem__, "foo")

def test_len():
    p = MockPopulation()
    assert_equal(len(p), MockPopulation.size)

def test_iter():
    p = MockPopulation()
    itr = p.__iter__()
    assert hasattr(itr, "next")
    assert_equal(len(list(itr)), 6)

def test_is_local():
    p1 = MockPopulation()
    p2 = MockPopulation()
    id_local = MockID("curly", parent=p1)
    id_nonlocal = MockID("larry", parent=p1)
    assert p1.is_local(id_local)
    assert not p1.is_local(id_nonlocal)
    assert_raises(AssertionError, p2.is_local, id_local)
    
def test_all():
    p = MockPopulation()
    itr = p.all()
    assert hasattr(itr, "next")
    assert_equal(len(list(itr)), 13)

def test_add():
    p1 = MockPopulation()
    p2 = MockPopulation()
    assembly = p1 + p2
    assert isinstance(assembly, populations.Assembly)
    assert_equal(assembly.populations, [p1, p2])
    
def test_get_cell_position():
    p = MockPopulation()
    id = MockID("larry", parent=p)
    assert_arrays_equal(p._get_cell_position(id), numpy.array([0,1,2]))
    id = MockID("moe", parent=p)
    assert_arrays_equal(p._get_cell_position(id), numpy.array([6,7,8]))
    
def test_set_cell_position():
    p = MockPopulation()
    id = MockID("larry", parent=p)
    p._set_cell_position(id, numpy.array([100,101,102]))
    assert_equal(p.positions[0,0], 100)
    assert_equal(p.positions[0,1], 3)

def test_get_cell_initial_value():
    p = MockPopulation()
    id = MockID("curly", parent=p)
    assert_equal(p._get_cell_initial_value(id, "foo"), 98)

def test_set_cell_initial_value():
    p = MockPopulation()
    id = MockID("curly", parent=p)
    p._set_cell_initial_value(id, "foo", -1)
    assert_equal(p._get_cell_initial_value(id, "foo"), -1)

def test_nearest():
    p = MockPopulation()
    p.positions = numpy.arange(39).reshape((13,3)).T
    assert_equal(p.nearest((0.0, 1.0, 2.0)), p[0])
    assert_equal(p.nearest((3.0, 4.0, 5.0)), p[1])
    assert_equal(p.nearest((36.0, 37.0, 38.0)), p[12])
    assert_equal(p.nearest((1.49, 2.49, 3.49)), p[0])
    assert_equal(p.nearest((1.51, 2.51, 3.51)), p[1])

def test_sample():
    orig_pv = populations.PopulationView
    populations.PopulationView = Mock()
    p = MockPopulation()
    rng = Mock()
    rng.permutation = Mock(return_value=numpy.array([7,4,8,12,0,3,9,1,2,11,5,10,6]))
    pv = p.sample(5, rng=rng)
    assert_arrays_equal(populations.PopulationView.call_args[0][1], numpy.array([7,4,8,12,0]))
    populations.PopulationView = orig_pv

def test_get_should_call_get_array_if_it_exists():
    p = MockPopulation()
    p._get_array = Mock()
    p.get("tau_m")
    p._get_array.assert_called_with("tau_m")

def test_get_with_no_get_array():
    orig_iter = MockPopulation.__iter__
    MockPopulation.__iter__ = Mock(return_value=iter([Mock()]))
    p = MockPopulation()
    values = p.get("i_offset")
    assert_equal(values[0]._name, "i_offset")
    MockPopulation.__iter__ = orig_iter

def test_get_with_gather():
    np_orig = MockPopulation._simulator.state.num_processes
    rank_orig = MockPopulation._simulator.state.mpi_rank
    gd_orig = recording.gather_dict
    MockPopulation._simulator.state.num_processes = 2
    MockPopulation._simulator.state.mpi_rank =  0
    def mock_gather_dict(D): # really hacky
        assert isinstance(D[0], list)
        D[1] = [i-1 for i in D[0]] + [D[0][-1] + 1]
        return D
    recording.gather_dict = mock_gather_dict
    
    p = MockPopulation()
    p._get_array = Mock(return_value=numpy.arange(11.0, 23.0, 2.0))
    assert_arrays_equal(p.get("tau_m", gather=True),
                        numpy.arange(10.0, 23.0))
    
    MockPopulation._simulator.state.num_processes = np_orig
    MockPopulation._simulator.state.mpi_rank = rank_orig
    recording.gather_dict = gd_orig

def test_set_from_dict():
    p = MockPopulation()
    p._set_array = Mock()
    p.set({'tau_m': 43.21})
    p._set_array.assert_called_with(**{'tau_m': 43.21})

def test_set_from_pair():
    p = MockPopulation()
    p._set_array = Mock()
    p.set('tau_m', 12.34)
    p._set_array.assert_called_with(**{'tau_m': 12.34})
         
def test_set_invalid_type():
    p = MockPopulation()
    assert_raises(errors.InvalidParameterValueError, p.set, 'foo', {})
    assert_raises(errors.InvalidParameterValueError, p.set, [1,2,3])
    assert_raises(errors.InvalidParameterValueError, p.set, 'foo', 'bar')
    assert_raises(errors.InvalidParameterValueError, p.set, {'foo': 'bar'})

def test_set_inconsistent_type():
    p = MockPopulation()
    p._set_array = Mock()
    assert_raises(errors.InvalidParameterValueError, p.set, 'tau_m', [12.34, 56.78])

def test_set_with_no_get_array():
    mock_cell = Mock()
    orig_iter = MockPopulation.__iter__
    MockPopulation.__iter__ = Mock(return_value=iter([mock_cell]))
    p = MockPopulation()
    values = p.set("i_offset", 0.1)
    mock_cell.set_parameters.assert_called_with(**{"i_offset": 0.1})
    MockPopulation.__iter__ = orig_iter

def test_set_with_list():
    p = MockPopulation()
    p._set_array = Mock()
    p.set('spike_times', range(10))
    p._set_array.assert_called_with(**{'spike_times': range(10)})
    
def test_tset_with_numeric_values():
    p = MockPopulation()
    p._set_array = Mock()
    tau_m = numpy.linspace(10.0, 20.0, num=p.size)
    p.tset("tau_m", tau_m)
    assert_arrays_equal(p._set_array.call_args[1]['tau_m'], tau_m[p._mask_local])

def test_tset_with_array_values():
    p = MockPopulation()
    p._set_array = Mock()
    spike_times = numpy.linspace(0.0, 1000.0, num=10*p.size).reshape((p.size,10))
    p.tset("spike_times", spike_times)
    call_args = p._set_array.call_args[1]['spike_times']
    assert_equal(call_args.shape, spike_times[p._mask_local].shape)
    assert_arrays_equal(call_args.flatten(),
                        spike_times[p._mask_local].flatten())
    
def test_tset_invalid_dimensions_2D():
    """Population.tset(): If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
    p = MockPopulation()
    array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
    assert_raises(errors.InvalidDimensionsError, p.tset, 'i_offset', array_in)

def test_tset_invalid_dimensions_1D():
    p = MockPopulation()
    tau_m = numpy.linspace(10.0, 20.0, num=p.size+1)
    assert_raises(errors.InvalidDimensionsError, p.tset, "tau_m", tau_m)

def test_rset():
    """Population.rset()"""
    p = MockPopulation()
    rd = Mock()
    rnums = numpy.arange(p.size)
    rd.next = Mock(return_value=rnums)
    p.tset = Mock()
    p.rset("cm", rd)
    rd.next.assert_called_with(**{'mask_local': False, 'n': p.size})
    call_args = p.tset.call_args
    assert_equal(call_args[0][0], 'cm')
    assert_arrays_equal(call_args[0][1], rnums)

def test_rset_with_native_rng():
    p = MockPopulation()
    p._native_rset = Mock()
    rd = Mock()
    rd.rng = random.NativeRNG()
    p.rset('tau_m', rd)
    p._native_rset.assert_called_with('tau_m', rd)

def test_initialize():
    p = MockPopulation()
    p.initial_values = {}
    p._set_initial_value_array = Mock()
    p.initialize('v', -65.0)
    assert_equal(p.initial_values['v'].evaluate(simplify=True), -65.0)
    p._set_initial_value_array.assert_called_with('v', -65.0)    

def test_initialize_random_distribution():
    p = MockPopulation()
    p.initial_values = {}
    p._set_initial_value_array = Mock()
    class MockRandomDistribution(random.RandomDistribution):
        def next(self, n, mask_local):
            return 42*numpy.ones(n)[mask_local]
    p.initialize('v', MockRandomDistribution())
    assert_arrays_equal(p.initial_values['v'].evaluate(simplify=True), 42*numpy.ones(p.local_size))
    #p._set_initial_value_array.assert_called_with('v', 42*numpy.ones(p.size)) 

def test_can_record():
    p = MockPopulation()
    p.celltype = MockStandardCell({})
    assert p.can_record('v')
    assert not p.can_record('foo')
    
def test__record():
    p = MockPopulation()
    p.recorders = {'v': Mock()}
    p._record('v')
    meth, args, kwargs = p.recorders['v'].method_calls[0]
    id_arr, = args
    assert_equal(meth, 'record')
    assert_arrays_equal(id_arr, p.all_cells)

def test__record_invalid_variable():
    p = MockPopulation()
    assert_raises(errors.RecordingError, p._record, 'foo')

#def test__record_int():
    #p = MockPopulation()
    #p.recorders = {'spikes': Mock()}
    #p._record('spikes', 5)
    #meth, args, kwargs = p.recorders['spikes'].method_calls[0]
    #id_arr, = args
    #assert_equal(meth, 'record')
    #assert_equal(id_arr.size, 5)

#def test__record_with_RNG():
    #p = MockPopulation()
    #p.recorders = {'v': Mock()}
    #rng = Mock()
    #rng.permutation = Mock(return_value=numpy.arange(p.size))
    #p._record('v', 5, rng)
    #meth, args, kwargs = p.recorders['v'].method_calls[0]
    #id_arr, = args
    #assert_equal(meth, 'record')
    #assert_equal(id_arr.size, 5)
    #rng.permutation.assert_called_with(p.all_cells)

#def test__record_list():
    #record_list = ['curly', 'larry', 'moe'] # should really check that record_list contains IDs
    #p = MockPopulation()
    #p.recorders = {'v': Mock()}
    #p._record('v', record_list)
    #meth, args, kwargs = p.recorders['v'].method_calls[0]
    #id_list, = args
    #assert_equal(meth, 'record')
    #assert_equal(id_list, record_list)
    
def test_invalid_record_from():
    p = MockPopulation()
    assert_raises(Exception, p._record, 'v', 4.2)
    
def test_spike_recording():
    p = MockPopulation()
    p._record = Mock()
    p.record("arg1")
    p._record.assert_called_with('spikes', "arg1")
    
def test_record_v():
    p = MockPopulation()
    p._record = Mock()
    p.record_v("arg1")
    p._record.assert_called_with('v', "arg1")

def test_record_gsyn():
    p = MockPopulation()
    p._record = Mock()
    p.record_gsyn("arg1")
    p._record.assert_called_with('gsyn', "arg1")

def test_printSpikes():
    p = MockPopulation()
    p.recorders = {'spikes': Mock()}
    p.record_filter = "arg4"
    p.printSpikes("arg1", "arg2", "arg3")
    meth, args, kwargs = p.recorders['spikes'].method_calls[0]
    assert_equal(meth, 'write')
    assert_equal(args, ("arg1", "arg2", "arg3", "arg4"))
    
def test_getSpikes():
    p = MockPopulation()
    p.recorders = {'spikes': Mock()}
    p.record_filter = "arg3"
    p.getSpikes("arg1", "arg2")
    meth, args, kwargs = p.recorders['spikes'].method_calls[0]
    assert_equal(meth, 'get')
    assert_equal(args, ("arg1", "arg2", "arg3"))

def test_print_v():
    p = MockPopulation()
    p.recorders = {'v': Mock()}
    p.record_filter = "arg4"
    p.print_v("arg1", "arg2", "arg3")
    meth, args, kwargs = p.recorders['v'].method_calls[0]
    assert_equal(meth, 'write')
    assert_equal(args, ("arg1", "arg2", "arg3", "arg4"))
    
def test_get_v():
    p = MockPopulation()
    p.recorders = {'v': Mock()}
    p.record_filter = "arg3"
    p.get_v("arg1", "arg2")
    meth, args, kwargs = p.recorders['v'].method_calls[0]
    assert_equal(meth, 'get')
    assert_equal(args, ("arg1", "arg2", "arg3"))
    
def test_print_gsyn():
    p = MockPopulation()
    p.recorders = {'gsyn': Mock()}
    p.record_filter = "arg4"
    p.print_gsyn("arg1", "arg2", "arg3")
    meth, args, kwargs = p.recorders['gsyn'].method_calls[0]
    assert_equal(meth, 'write')
    assert_equal(args, ("arg1", "arg2", "arg3", "arg4"))
    
def test_get_gsyn():
    p = MockPopulation()
    p.recorders = {'gsyn': Mock()}
    p.record_filter = "arg3"
    p.get_gsyn("arg1", "arg2")
    meth, args, kwargs = p.recorders['gsyn'].method_calls[0]
    assert_equal(meth, 'get')
    assert_equal(args, ("arg1", "arg2", "arg3"))
    
def test_get_spike_counts():
    p = MockPopulation()
    p.recorders = {'spikes': Mock()}
    p.get_spike_counts("arg1")
    meth, args, kwargs = p.recorders['spikes'].method_calls[0]
    assert_equal(meth, 'count')
    assert_equal(args, ("arg1", None))
    
def test_mean_spike_count():
    orig_rank = MockPopulation._simulator.state.mpi_rank
    MockPopulation._simulator.state.mpi_rank = 0
    p = MockPopulation()
    p.recorders = {'spikes': Mock()}
    p.recorders['spikes'].count = Mock(return_value={0: 2, 1: 5})
    assert_equal(p.mean_spike_count(), 3.5)
    MockPopulation._simulator.state.mpi_rank = orig_rank

def test_mean_spike_count_on_slave_node():
    orig_rank = MockPopulation._simulator.state.mpi_rank
    MockPopulation._simulator.state.mpi_rank = 1
    p = MockPopulation()
    p.recorders = {'spikes': Mock()}
    p.recorders['spikes'].count = Mock(return_value={0: 2, 1: 5})
    assert p.mean_spike_count() is numpy.NaN
    MockPopulation._simulator.state.mpi_rank = orig_rank
    
def test_inject():
    p = MockPopulation()
    cs = Mock()
    p.inject(cs)
    meth, args, kwargs = cs.method_calls[0]
    assert_equal(meth, "inject_into")
    assert_equal(args, (p,))

def test_inject_into_invalid_celltype():
    p = MockPopulation()
    p.celltype.injectable = False
    assert_raises(TypeError, p.inject, Mock())

def test_save_positions():
    import os
    orig_rank = MockPopulation._simulator.state.mpi_rank
    MockPopulation._simulator.state.mpi_rank = 0
    p = MockPopulation()
    p.all_cells = numpy.array([34, 45, 56, 67])
    p.positions = numpy.arange(12).reshape((4,3)).T
    output_file = Mock()
    p.save_positions(output_file)
    assert_arrays_equal(output_file.write.call_args[0][0],
                        numpy.array([[34, 0, 1, 2], [45, 3, 4, 5], [56, 6, 7, 8], [67, 9, 10, 11]]))
    assert_equal(output_file.write.call_args[0][1], {'population': p.label})
    # arguably, the first column should contain indices, not ids.
    MockPopulation._simulator.state.mpi_rank = orig_rank