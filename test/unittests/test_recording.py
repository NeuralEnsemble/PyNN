from pyNN import recording
from nose.tools import assert_equal, assert_raises
from mock import Mock
import numpy
import os
from pyNN.utility import assert_arrays_equal

MPI = recording.MPI
if MPI:
    mpi_comm = recording.mpi_comm

def setup():
    recording.Recorder._simulator = MockSimulator(mpi_rank=0)
    
def teardown():
    del recording.Recorder._simulator

#def test_rename_existing():
    
#def test_gather():
    #import time
    #for x in range(7):
    #    N = pow(10, x)
    #    local_data = numpy.empty((N,2))
    #    local_data[:,0] = numpy.ones(N, dtype=float)*comm.rank
    #    local_data[:,1] = numpy.random.rand(N)
    #    
    #    start_time = time.time()
    #    all_data = gather(local_data)
    #    #print comm.rank, "local", local_data
    #    if comm.rank == 0:
    #    #    print "all", all_data
    #        print N, time.time()-start_time
    
#def test_gather_no_MPI():

#def test_gather_dict():

#def test_mpi_sum():

class MockPopulation(object):
    size = 11
    first_id = 2454
    last_id = first_id + size
    label = "mock population"
    def __len__(self):
        return self.size
    def can_record(self, variable):
        if variable in ["spikes", "v", "gsyn"]:
            return True
        else:
            return False
    def id_to_index(self, id):
        return id


def test_Recorder_create():
    r = recording.Recorder('spikes')
    assert_equal(r.variable, 'spikes')
    assert_equal(r.population, None)
    assert_equal(r.file, None)
    assert_equal(r.recorded, set([]))
    
def test_Recorder_invalid_variable():
    assert_raises(AssertionError,
                  recording.Recorder, 'foo', population=MockPopulation())

class MockID(object):
    def __init__(self, id, local):
        self.id = id
        self.local = local

def test_record():
    r = recording.Recorder('spikes')
    r._record = Mock()
    assert_equal(r.recorded, set([]))
    
    all_ids = (MockID(0, True), MockID(1, False), MockID(2, True), MockID(3, True), MockID(4, False))
    first_ids = all_ids[0:3]
    r.record(first_ids)
    assert_equal(r.recorded, set(id for id in first_ids if id.local))
    assert_equal(len(r.recorded), 2)
    r._record.assert_called_with(r.recorded)
    
    more_ids = all_ids[2:5]
    r.record(more_ids)
    assert_equal(r.recorded, set(id for id in all_ids if id.local))
    assert_equal(len(r.recorded), 3)
    r._record.assert_called_with(set(all_ids[3:4]))

def test_filter_recorded():
    r = recording.Recorder('spikes')
    r._record = Mock()
    all_ids = (MockID(0, True), MockID(1, False), MockID(2, True), MockID(3, True), MockID(4, False))
    r.record(all_ids)
    assert_equal(r.recorded, set(id for id in all_ids if id.local))

    filter = all_ids[::2]
    filtered_ids = r.filter_recorded(filter)
    assert_equal(filtered_ids, set(id for id in filter if id.local))
    
    assert_equal(r.filter_recorded(None), r.recorded)

def test_get__zero_offset():
    r = recording.Recorder('spikes')
    fake_data = numpy.array([
                    (3, 12.3),
                    (4, 14.5),
                    (7, 19.8)
                ])
    r._get = Mock(return_value=fake_data)
    assert_arrays_equal(r.get(), fake_data)


class MockState(object):
    def __init__(self, mpi_rank):
        self.mpi_rank = mpi_rank
        self.dt = 0.123
class MockSimulator(object):
    def __init__(self, mpi_rank):
        self.state = MockState(mpi_rank)

def test_write__with_filename__compatible_output__gather__onroot():
    orig_metadata = recording.Recorder.metadata
    recording.Recorder.metadata = {'a': 2, 'b':3}
    r = recording.Recorder('spikes')
    fake_data = numpy.array([
                    (3, 12.3),
                    (4, 14.5),
                    (7, 19.8)
                ])
    r._get = Mock(return_value=fake_data)
    r._make_compatible = Mock(return_value=fake_data)
    r.write(file="tmp.spikes", gather=True, compatible_output=True)

    os.remove("tmp.spikes")
    recording.Recorder.metadata = orig_metadata

def test_metadata_property():
    r = recording.Recorder('spikes', population=None)
    r._get = Mock(return_value=numpy.random.uniform(size=(6,2)))
    assert_equal(r.metadata,
                 {'variable': 'spikes', 'dt': 0.123, 'n': 6})
    
    r = recording.Recorder('v', population=MockPopulation())
    r._get = Mock(return_value=numpy.random.uniform(size=(6,2)))
    assert_equal(r.metadata,
                 {'first_id': 2454, 'label': 'mock population', 'n': 6,
                  'variable': 'v', 'dt': 0.123, 'last_id': 2465, 'size': 11,
                  'first_index': 0, 'last_index': 11})
    
def test__make_compatible_spikes():
    r = recording.Recorder('spikes')
    input_data = numpy.array([[0, 12.3], [1, 45.2], [0, 46.3],
                              [4, 49.4], [0, 78.3]])
    output_data = r._make_compatible(input_data) # time id
    assert_arrays_equal(input_data[:,(1,0)], output_data)

def test__make_compatible_v():
    r = recording.Recorder('v')
    input_data = numpy.array([[0, 0.0, -65.0], [3, 0.0, -65.0],
                              [0, 0.1, -64.3], [3, 0.1, -65.1],
                              [0, 0.2, -63.7], [3, 0.2, -65.5]])
    output_data = r._make_compatible(input_data) # voltage id
    assert_arrays_equal(input_data[:,(2,0)], output_data) 

#def test_count__spikes_gather():

#def test_count__spikes_nogather():


#def test_count__other():

