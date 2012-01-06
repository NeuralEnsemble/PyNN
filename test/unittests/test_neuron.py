from neuron import h
from pyNN.common import populations
from pyNN.neuron.standardmodels import electrodes
from pyNN.neuron import recording, simulator
from mock import Mock
from nose.tools import assert_equal, assert_raises, assert_almost_equal
import numpy

class MockCellClass(object):
    recordable = ['v']
    parameters = ['romans', 'judeans']
    injectable = True
    @classmethod
    def has_parameter(cls, name):
        return name in cls.parameters

class MockCell(object):
    parameter_names = ['romans', 'judeans']
    def __init__(self, romans=0, judeans=1):
        self.source_section = h.Section()
        self.source = self.source_section(0.5)._ref_v
        self.synapse = h.tmgsyn(self.source_section(0.5))
        self.record = Mock()
        self.record_v = Mock()
        self.record_gsyn = Mock()
        self.memb_init = Mock()
        self.excitatory_TM = None
        self.inhibitory_TM = None
        self.romans = romans
        self.judeans = judeans
        self.foo_init = -99.9

class MockStepCurrentSource(object):
    parameter_names = ['amplitudes', 'times']
    def __init__(self, parameters):
        self._devices = []

    def inject_into(self, cell_list):
        for cell in cell_list:
            if cell.local:
               self._devices += [cell]

class MockDCSource(object):
    parameter_names = ['amplitude', 'start', 'stop']
    def __init__(self, parameters):
        self._devices = []

    def inject_into(self, cell_list):
        for cell in cell_list:
            if cell.local:
               self._devices += [cell]


class MockID(int):
    def __init__(self, n):
        int.__init__(n)
        self.local = bool(n%2)
        self.celltype = MockCellClass()
        self._cell = MockCell()

class MockPopulation(populations.BasePopulation):
    celltype = MockCellClass()
    local_cells = [MockID(44), MockID(33)]

# simulator
def test_load_mechanisms():
    assert_raises(Exception, simulator.load_mechanisms, "/tmp") # not found
    
def test_is_point_process():
    section = h.Section()
    clamp = h.SEClamp(section(0.5))
    assert simulator.is_point_process(clamp)
    section.insert('hh')
    assert not simulator.is_point_process(section(0.5).hh)

def test_native_rng_pick():
    rng = Mock()
    rng.seed = 28754
    rarr = simulator.nativeRNG_pick(100, rng, 'uniform', [-3,6])
    assert isinstance(rarr, numpy.ndarray)
    assert_equal(rarr.shape, (100,))
    assert -3 <= rarr.min() < -2.5
    assert 5.5 < rarr.max() < 6

def test_register_gid():
    cell = MockCell()
    simulator.register_gid(84568345, cell.source, cell.source_section)

class TestInitializer(object):

    def test_initializer_initialize(self):
        init = simulator.initializer
        orig_initialize = init._initialize
        init._initialize = Mock()
        h.finitialize(-65)
        init._initialize.assert_called()
        init._initialize = orig_initialize
    
    def test_register(self):
        init = simulator.initializer
        cell = MockID(22)
        pop = MockPopulation()
        init.clear()
        init.register(cell, pop)
        assert_equal(init.cell_list, [cell])
        assert_equal(init.population_list, [pop])

    def test_initialize(self):
        init = simulator.initializer
        cell = MockID(77)
        pop = MockPopulation()
        init.register(cell, pop)
        init._initialize()
        cell._cell.memb_init.assert_called()
        for pcell in pop.local_cells:
            pcell._cell.memb_init.assert_called()

    def test_clear(self):
        init = simulator.initializer
        init.cell_list = range(10)
        init.population_list = range(10)
        init.clear()
        assert_equal(init.cell_list, [])
        assert_equal(init.population_list, [])


class TestState(object):
    
    def test_dt_property(self):
        simulator.state.dt = 0.01
        assert_equal(h.dt, 0.01)
        assert_equal(h.steps_per_ms, 100.0)
        assert_equal(simulator.state.dt, 0.01)


def test_reset():
    simulator.state.running = True
    simulator.state.t = 17
    simulator.state.tstop = 123
    init = simulator.initializer
    orig_initialize = init._initialize
    init._initialize = Mock()
    simulator.reset()
    assert_equal(simulator.state.running, False)
    assert_equal(simulator.state.t, 0.0)
    assert_equal(simulator.state.tstop, 0.0)
    init._initialize.assert_called()
    init._initialize = orig_initialize


def test_run():
    simulator.reset()
    simulator.run(12.3)
    assert_almost_equal(h.t, 12.3, places=11)
    simulator.run(7.7)
    assert_almost_equal(h.t, 20.0, places=11)

def test_finalize():
    orig_pc = simulator.state.parallel_context
    simulator.state.parallel_context = Mock()
    simulator.finalize()
    simulator.state.parallel_context.runworker.assert_called()
    simulator.state.parallel_context.done.assert_called()
    simulator.state.parallel_context = orig_pc


class TestID(object):
    
    def setup(self):
        self.id = simulator.ID(984329856)
        self.id.parent = MockPopulation()
        self.id._cell = MockCell()

    def test_create(self):
        assert_equal(self.id, 984329856)

    def test_build_cell(self):
        parameters = {'judeans': 1, 'romans': 0}
        self.id._build_cell(MockCell, parameters)
     
    def test_get_native_parameters(self):   
        D = self.id.get_native_parameters()
        assert isinstance(D, dict)
    
    def test_set_native_parameters(self):   
        self.id.set_native_parameters({'romans': 3, 'judeans': 1})
    
    def test_get_initial_value(self):
        foo_init = self.id.get_initial_value('foo')
        assert_equal(foo_init, -99.9)
        
    #def test_set_initial_value(self):
        
        
class TestConnection(object):
    
    def setup(self):
        self.source = MockID(252)
        self.target = MockID(539)
        self.nc = h.NetCon(self.source._cell.source,
                           self.target._cell.synapse,
                           sec=self.source._cell.source_section)
        self.nc.delay = 1.0
        self.c = simulator.Connection(self.source, self.target, self.nc)
    
    def test_create(self):
        c = self.c
        assert_equal(c.source, self.source)
        assert_equal(c.target, self.target)
    
    def test_useSTDP(self):
        self.c.useSTDP("StdwaSA", {'wmax': 0.04}, ddf=0)

    def test_weight_property(self):
        self.nc.weight[0] = 0.123
        assert_equal(self.c.weight, 0.123)
        self.c.weight = 0.234
        assert_equal(self.nc.weight[0], 0.234)
        
    def test_delay_property(self):
        self.nc.delay = 12.3
        assert_equal(self.c.delay, 12.3)
        self.c.delay = 23.4
        assert_equal(self.nc.delay, 23.4)

    def test_U_property(self):
        self.target._cell.synapse.U = 0.1
        assert_equal(self.c.U, 0.1)
        self.c.U = 0.2
        assert_equal(self.target._cell.synapse.U, 0.2)

    def test_w_max_property(self):
        self.c.useSTDP("StdwaSA", {'wmax': 0.04}, ddf=0)
        assert_equal(self.c.w_max, 0.04)
        self.c.w_max = 0.05
        assert_equal(self.c.weight_adjuster.wmax, 0.05)


# electrodes
class TestCurrentSources(object):

    def setup(self):
        self.cells = [MockID(n) for n in range(5)]

    def test_inject_dc(self):
        cs = electrodes.DCSource({})
        cs.inject_into(self.cells)
        assert_equal(cs.stop, 1e12)
        assert_equal(len(cs._devices), 2) 

    def test_inject_step_current(self):
        cs = MockStepCurrentSource({'amplitudes' : [1,2,3], 'times' : [0.5, 1.5, 2.5]})
        cs.inject_into(self.cells)
        assert_equal(len(cs._devices), 2)# 2 local cells
        # need more assertions about iclamps, vectors
        
        
# recording
class TestRecorder(object):
    
    def setup(self):
        if "foo" not in recording.Recorder.formats:
            recording.Recorder.formats['foo'] = "bar"
        self.rv = recording.Recorder('v')
        self.rg = recording.Recorder('gsyn')
        self.rs = recording.Recorder('spikes')
        self.rf = recording.Recorder('foo')
        self.cells = [MockID(22), MockID(29)]
    
    def teardown(self):
        recording.Recorder.formats.pop("foo")
    
    def test__record(self):
        self.rv._record(self.cells)
        self.rg._record(self.cells)
        self.rs._record(self.cells)
        for cell in self.cells:
            cell._cell.record.assert_called_with(1)
            cell._cell.record_v.assert_called_with(1)
            cell._cell.record_gsyn.assert_called_with('inhibitory', 1)
        assert_raises(Exception, self.rf._record, self.cells)
        
    def test__get_v(self):
        self.rv.recorded = self.cells
        self.cells[0]._cell.vtrace = numpy.arange(-65.0, -64.0, 0.1)
        self.cells[1]._cell.vtrace = numpy.arange(-64.0, -65.0, -0.1)
        self.cells[0]._cell.record_times = self.cells[1]._cell.record_times = numpy.arange(0.0, 1.0, 0.1)
        vdata = self.rv._get(gather=False, compatible_output=True, filter=None)
        assert_equal(vdata.shape, (20,3))
        
    def test__get_spikes(self):
        self.rs.recorded = self.cells
        self.cells[0]._cell.spike_times = numpy.arange(101.0, 111.0)
        self.cells[1]._cell.spike_times = numpy.arange(13.0, 23.0)
        simulator.state.t = 111.0
        sdata = self.rs._get(gather=False, compatible_output=True, filter=None)
        assert_equal(sdata.shape, (20,2))
        
    def test__get_gsyn(self):
        self.rg.recorded = self.cells
        for cell in self.cells:
            cell._cell.gsyn_trace = {}
            cell._cell.gsyn_trace['excitatory'] = numpy.arange(0.01, 0.0199, 0.001)
            cell._cell.gsyn_trace['inhibitory'] = numpy.arange(1.01, 1.0199, 0.001)
            cell._cell.gsyn_trace['excitatory_TM'] = numpy.arange(2.01, 2.0199, 0.001)
            cell._cell.gsyn_trace['inhibitory_TM'] = numpy.arange(4.01, 4.0199, 0.001)
            cell._cell.record_times = self.cells[1]._cell.record_times = numpy.arange(0.0, 1.0, 0.1)
        gdata = self.rg._get(gather=False, compatible_output=True, filter=None)
        assert_equal(gdata.shape, (20,4))
    
    def test__local_count(self):
        self.rs.recorded = self.cells
        self.cells[0]._cell.spike_times = h.Vector(numpy.arange(101.0, 111.0))
        self.cells[1]._cell.spike_times = h.Vector(numpy.arange(13.0, 33.0))
        assert_equal(self.rs._local_count(filter=None), {22: 10, 29: 20})
    
