from pyNN import recording, errors
from nose.tools import assert_equal, assert_raises
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
import numpy
import os
from datetime import datetime
from collections import defaultdict
from pyNN.utility import assert_arrays_equal

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
    #    #print(comm.rank, "local", local_data)
    #    if comm.rank == 0:
    #    #    print("all", all_data)
    #        print(N, time.time()-start_time)
    
#def test_gather_no_MPI():

#def test_gather_dict():

#def test_mpi_sum():

class MockState(object):
    def __init__(self, mpi_rank):
        self.mpi_rank = mpi_rank
        self.num_processes = 9
        self.dt = 0.123
        self.running = True
        self.recorders = set([])
        self.t = 0.0
class MockSimulator(object):
    name = "MockSimulator"
    def __init__(self, mpi_rank):
        self.state = MockState(mpi_rank)

class MockNeoIO(object):
    filename = "fake_file"
    write = Mock()

class MockRecorder(recording.Recorder):
    _simulator = MockSimulator(mpi_rank=0)

class MockPopulation(object):
    size = 11
    first_id = 2454
    last_id = first_id + size
    label = "mock population"
    celltype = Mock(always_local=False)
    annotations = {'knights_say': 'Ni!'}
    def __len__(self):
        return self.size
    def can_record(self, variable):
        if variable in ["spikes", "v", "gsyn_exc", "gsyn_inh", "spam"]:
            return True
        else:
            return False
    def id_to_index(self, id):
        return id
    def describe(self):
        return "mock population"

class MockNeoBlock(object):
    def __init__(self):
        self.name = None
        self.description = None
        self.segments = [Mock()]
        self.rec_datetime = datetime.now()
    def annotate(self, **annotations):
        pass

def test_Recorder_create():
    p = MockPopulation()
    r = MockRecorder(p)
    assert_equal(r.population, p)
    assert_equal(r.file, None)
    assert_equal(r.recorded, defaultdict(set))
    
def test_Recorder_invalid_variable():
    p = MockPopulation()
    r = MockRecorder(p)
    all_ids = (MockID(0, True), MockID(1, False), MockID(2, True), MockID(3, True), MockID(4, False))
    assert_raises(errors.RecordingError,
                  r.record, 'foo', all_ids)

class MockID(object):
    def __init__(self, id, local):
        self.id = id
        self.local = local

def test_record():
    p = MockPopulation()
    r = MockRecorder(p)
    r._record = Mock()
    assert_equal(r.recorded, defaultdict(set))
    
    all_ids = (MockID(0, True), MockID(1, False), MockID(2, True), MockID(3, True), MockID(4, False))
    first_ids = all_ids[0:3]
    r.record('spam', first_ids)
    assert_equal(r.recorded['spam'], set(id for id in first_ids if id.local))
    assert_equal(len(r.recorded['spam']), 2)
    r._record.assert_called_with('spam', r.recorded['spam'], None)
    
    more_ids = all_ids[2:5]
    r.record('spam', more_ids)
    assert_equal(r.recorded['spam'], set(id for id in all_ids if id.local))
    assert_equal(len(r.recorded['spam']), 3)
    r._record.assert_called_with('spam', set(all_ids[3:4]), None)

def test_filter_recorded():
    p = MockPopulation()
    r = MockRecorder(p)
    r._record = Mock()
    all_ids = (MockID(0, True), MockID(1, False), MockID(2, True), MockID(3, True), MockID(4, False))
    r.record(['spikes', 'spam'], all_ids)
    assert_equal(r.recorded['spikes'], set(id for id in all_ids if id.local))
    assert_equal(r.recorded['spam'], set(id for id in all_ids if id.local))

    filter = all_ids[::2]
    filtered_ids = r.filter_recorded('spam', filter)
    assert_equal(filtered_ids, set(id for id in filter if id.local))
    
    assert_equal(r.filter_recorded('spikes', None), r.recorded['spikes'])

def test_get():
    p = MockPopulation()
    r = MockRecorder(p)
    r._get_current_segment = Mock()
    data = r.get('spikes')
    assert_equal(data.name, p.label)
    assert_equal(data.description, p.describe())

#def test_write__with_filename__compatible_output__gather__onroot():
#    orig_metadata = recording.Recorder.metadata
#    #recording.Recorder.metadata = {'a': 2, 'b':3}
#    p = MockPopulation()
#    r = MockRecorder(p)
#    #fake_data = MockNeoBlock()
#    r._get_current_segment = Mock() #return_value=fake_data)
#    output_io = MockNeoIO()
#    r.write("spikes", file=output_io, gather=True)
#    #recording.Recorder.metadata = orig_metadata
#    output_io.write.assert_called_with(fake_data)

def test_metadata_property():
    p = MockPopulation()
    r = MockRecorder(p)
    assert_equal(r.metadata,
                 {'first_id': 2454, 'label': 'mock population',
                  'dt': 0.123, 'last_id': 2465, 'size': 11,
                  'first_index': 0, 'last_index': 11, 'knights_say': 'Ni!',
                  'simulator': 'MockSimulator', 'mpi_processes': 9})


#def test_count__spikes_gather():

#def test_count__spikes_nogather():


#def test_count__other():

