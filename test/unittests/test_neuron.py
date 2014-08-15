# encoding: utf-8

import os
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
try:
    basestring
except NameError:
    basestring = str
try:
    from neuron import h
    import pyNN.neuron as sim
    from pyNN.neuron.standardmodels import electrodes
    from pyNN.neuron import recording, simulator, cells
except ImportError:
    sim = False
    h = Mock()
    
from pyNN.common import populations
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal


skip_ci = False
if "JENKINS_SKIP_TESTS" in os.environ:
    skip_ci = os.environ["JENKINS_SKIP_TESTS"] == "1"


class MockCellClass(object):
    recordable = ['v', 'spikes', 'gsyn_exc', 'gsyn_inh', 'spam']
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
        #self.synapse = h.tmgsyn(self.source_section(0.5))
        self.record = Mock()
        self.record_v = Mock()
        self.record_gsyn = Mock()
        self.memb_init = Mock()
        self.excitatory = h.ExpSyn(self.source_section(0.5))
        self.inhibitory = None
        self.romans = romans
        self.judeans = judeans
        self.foo_init = -99.9
        self.traces = {}

    def __call__(self, pos):
        return Mock()


class MockSynapseType(object):
    model = None

class MockPlasticSynapseType(object):
    model = "StdwaSA"
    postsynaptic_variable = "spikes"

class MockStepCurrentSource(object):
    parameter_names = ['amplitudes', 'times']
    def __init__(self, **parameters):
        self._devices = []

    def inject_into(self, cell_list):
        for cell in cell_list:
            if cell.local:
               self._devices += [cell]

class MockDCSource(object):
    parameter_names = ['amplitude', 'start', 'stop']
    def __init__(self, **parameters):
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
    all_cells = local_cells
    label = "mock population"
    def describe(self):
        return "mock population"

class MockProjection(object):
    receptor_type = 'excitatory'
    synapse_type = MockSynapseType()
    pre = MockPopulation()
    post = MockPopulation()
    

@unittest.skipUnless(sim, "Requires NEURON")
class TestFunctions(unittest.TestCase):

    def test_load_mechanisms(self):
        self.assertRaises(Exception, simulator.load_mechanisms, "/tmp") # not found
    
    def test_is_point_process(self):
        section = h.Section()
        clamp = h.SEClamp(section(0.5))
        assert simulator.is_point_process(clamp)
        section.insert('hh')
        assert not simulator.is_point_process(section(0.5).hh)

    def test_native_rng_pick(self):
        rng = Mock()
        rng.seed = 28754
        rarr = simulator.nativeRNG_pick(100, rng, 'uniform', [-3,6])
        assert isinstance(rarr, numpy.ndarray)
        self.assertEqual(rarr.shape, (100,))
        assert -3 <= rarr.min() < -2.5
        assert 5.5 < rarr.max() < 6

    def test_list_standard_models(self):
        cell_types = sim.list_standard_models()
        self.assertTrue(len(cell_types) > 10)
        self.assertIsInstance(cell_types[0], basestring)

    def test_setup(self):
        sim.setup(timestep=0.05, min_delay=0.1, max_delay=1.0)
        self.assertEqual(h.dt, 0.05)
        # many more things could be tested here

    def test_setup_with_cvode(self):
        sim.setup(timestep=0.05, min_delay=0.1, max_delay=1.0,
                  use_cvode=True, rtol=1e-2, atol=2e-6)
        self.assertEqual(h.dt, 0.05)
        self.assertEqual(simulator.state.cvode.rtol(), 1e-2)
        # many more things could be tested here


@unittest.skipUnless(sim, "Requires NEURON")
class TestInitializer(unittest.TestCase):

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
        self.assertEqual(init.cell_list, [cell])
        self.assertEqual(init.population_list, [pop])

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
        self.assertEqual(init.cell_list, [])
        self.assertEqual(init.population_list, [])


@unittest.skipUnless(sim, "Requires NEURON")
class TestState(unittest.TestCase):

    def test_register_gid(self):
        cell = MockCell()
        simulator.state.register_gid(84568345, cell.source, cell.source_section)


    def test_dt_property(self):
        simulator.state.dt = 0.01
        self.assertEqual(h.dt, 0.01)
        self.assertEqual(h.steps_per_ms, 100.0)
        self.assertEqual(simulator.state.dt, 0.01)

    #def test_reset(self):
    #    simulator.state.running = True
    #    simulator.state.t = 17
    #    simulator.state.tstop = 123
    #    init = simulator.initializer
    #    orig_initialize = init._initialize
    #    init._initialize = Mock()
    #    simulator.state.reset()
    #    self.assertEqual(simulator.state.running, False)
    #    self.assertEqual(simulator.state.t, 0.0)
    #    self.assertEqual(simulator.state.tstop, 0.0)
    #    init._initialize.assert_called()
    #    init._initialize = orig_initialize

    #def test_run(self):
    #    simulator.state.reset()
    #    simulator.state.run(12.3)
    #    self.assertAlmostEqual(h.t, 12.3, places=11)
    #    simulator.state.run(7.7)
    #    self.assertAlmostEqual(h.t, 20.0, places=11)

    def test_finalize(self):
        orig_pc = simulator.state.parallel_context
        simulator.state.parallel_context = Mock()
        simulator.state.finalize()
        simulator.state.parallel_context.runworker.assert_called()
        simulator.state.parallel_context.done.assert_called()
        simulator.state.parallel_context = orig_pc


@unittest.skipUnless(sim, "Requires NEURON")
class TestPopulation(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.p = sim.Population(4, sim.IF_cond_exp(**{'tau_m': 12.3,
                                                      'cm': lambda i: 0.987 + 0.01*i,
                                                      'i_offset': numpy.array([-0.21, -0.20, -0.19, -0.18])}))

    def test__get_parameters(self):
        ps = self.p._get_parameters('c_m', 'tau_m', 'e_e', 'i_offset')
        ps.evaluate(simplify=True)
        assert_array_almost_equal(ps['c_m'], numpy.array([0.987, 0.997, 1.007, 1.017], float),
                                  decimal=12)
        assert_array_almost_equal(ps['i_offset'], numpy.array([-0.21, -0.2, -0.19, -0.18], float),
                                  decimal=12)
        self.assertEqual(ps['e_e'], 0.0)


@unittest.skipUnless(sim, "Requires NEURON")
@unittest.skipIf(skip_ci, "Skipping test on CI server")
class TestID(unittest.TestCase):

    def setUp(self):
        self.id = simulator.ID(984329856)
        self.id.parent = MockPopulation()
        self.id._cell = MockCell()

    def test_create(self):
        self.assertEqual(self.id, 984329856)

    def test_build_cell(self):
        parameters = {'judeans': 1, 'romans': 0}
        self.id._build_cell(MockCell, parameters)

    def test_get_initial_value(self):
        foo_init = self.id.get_initial_value('foo')
        self.assertEqual(foo_init, -99.9)

    #def test_set_initial_value(self):


@unittest.skipUnless(sim, "Requires NEURON")
class TestConnection(unittest.TestCase):

    def setUp(self):
        self.pre = 0
        self.post = 1
        self.c = simulator.Connection(MockProjection(), self.pre, self.post,
                                      weight=0.123, delay=0.321)

    def test_create(self):
        c = self.c
        self.assertEqual(c.presynaptic_index, self.pre)
        self.assertEqual(c.postsynaptic_index, self.post)

    def test_setup_plasticity(self):
        self.c._setup_plasticity(MockPlasticSynapseType(),
                                 {'wmax': 0.04,
                                  'dendritic_delay_fraction': 0.234})

    def test_weight_property(self):
        self.c.nc.weight[0] = 0.123
        self.assertEqual(self.c.weight, 0.123)
        self.c.weight = 0.234
        self.assertEqual(self.c.nc.weight[0], 0.234)

    def test_delay_property(self):
        self.c.nc.delay = 12.3
        self.assertEqual(self.c.delay, 12.3)
        self.c.delay = 23.4
        self.assertEqual(self.c.nc.delay, 23.4)

    def test_w_max_property(self):
        self.c._setup_plasticity(MockPlasticSynapseType(),
                                 {'wmax': 0.04,
                                  'dendritic_delay_fraction': 0})
        self.assertEqual(self.c.w_max, 0.04)
        self.c.w_max = 0.05
        self.assertEqual(self.c.weight_adjuster.wmax, 0.05)


@unittest.skipUnless(sim, "Requires NEURON")
class TestProjection(unittest.TestCase):

    def setUp(self):
        sim.setup()
        self.p1 = sim.Population(7, sim.IF_cond_exp())
        self.p2 = sim.Population(4, sim.IF_cond_exp())
        self.p3 = sim.Population(5, sim.IF_curr_alpha())
        self.syn1 = sim.StaticSynapse(weight=0.123, delay=0.5)
        self.syn2 = sim.StaticSynapse(weight=0.456, delay=0.4)
        self.random_connect = sim.FixedNumberPostConnector(n=2)
        self.all2all = sim.AllToAllConnector()

    def test_create_simple(self):
        prj = sim.Projection(self.p1, self.p2, self.all2all, self.syn2)

    def test_create_with_fast_synapse_dynamics(self):
        prj = sim.Projection(self.p1, self.p2, self.all2all,
                             synapse_type=sim.TsodyksMarkramSynapse())


@unittest.skipUnless(sim, "Requires NEURON")
class TestCurrentSources(unittest.TestCase):

    def setUp(self):
        self.cells = [MockID(n) for n in range(5)]

    def test_inject_dc(self):
        cs = electrodes.DCSource()
        cs.inject_into(self.cells)
        self.assertEqual(cs.stop, 1e12)
        self.assertEqual(len(cs._devices), 2)

    def test_inject_step_current(self):
        cs = MockStepCurrentSource(amplitudes=[1,2,3], times=[0.5, 1.5, 2.5])
        cs.inject_into(self.cells)
        self.assertEqual(len(cs._devices), 2)# 2 local cells
        # need more assertions about iclamps, vectors


@unittest.skipUnless(sim, "Requires NEURON")
class TestRecorder(unittest.TestCase):

    def setUp(self):
        self.p = sim.Population(2, sim.IF_cond_exp())
        self.rec = recording.Recorder(self.p)
        self.cells = self.p.all_cells #[MockID(22), MockID(29)]

    def tearDown(self):
        pass

    def test__record(self):
        self.rec._record('v', self.cells)
        self.rec._record('gsyn_inh', self.cells)
        self.rec._record('spikes', self.cells)
        self.assertRaises(Exception, self.rec._record, self.cells)
    
    #def test__get_v(self):
    #    self.rv.recorded['v'] = self.cells
    #    self.cells[0]._cell.vtrace = numpy.arange(-65.0, -64.0, 0.1)
    #    self.cells[1]._cell.vtrace = numpy.arange(-64.0, -65.0, -0.1)
    #    self.cells[0]._cell.record_times = self.cells[1]._cell.record_times = numpy.arange(0.0, 1.0, 0.1)
    #    simulator.state.t = simulator.state.dt * len(self.cells[0]._cell.vtrace)
    #    vdata = self.rv._get_current_segment(variables=['v'], filter_ids=None)
    #    self.assertEqual(len(vdata.analogsignalarrays), 1)
    #    assert_array_equal(numpy.array(vdata.analogsignalarrays[0]),
    #                        numpy.vstack((self.cells[0]._cell.vtrace, self.cells[1]._cell.vtrace)).T)

    def test__get_spikes(self):
        self.rec.recorded['spikes'] = self.cells
        self.cells[0]._cell.spike_times = numpy.arange(101.0, 111.0)
        self.cells[1]._cell.spike_times = numpy.arange(13.0, 23.0)
        simulator.state.t = 111.0
        sdata = self.rec._get_current_segment(variables=['spikes'], filter_ids=None)
        self.assertEqual(len(sdata.spiketrains), 2)
        assert_array_equal(numpy.array(sdata.spiketrains[0]), self.cells[0]._cell.spike_times)

    #def test__get_gsyn(self):
    #    self.rg.recorded['gsyn_exc'] = self.cells
    #    self.rg.recorded['gsyn_inh'] = self.cells
    #    for cell in self.cells:
    #        cell._cell.gsyn_trace = {}
    #        cell._cell.gsyn_trace['excitatory'] = numpy.arange(0.01, 0.0199, 0.001)
    #        cell._cell.gsyn_trace['inhibitory'] = numpy.arange(1.01, 1.0199, 0.001)
    #        cell._cell.gsyn_trace['excitatory_TM'] = numpy.arange(2.01, 2.0199, 0.001)
    #        cell._cell.gsyn_trace['inhibitory_TM'] = numpy.arange(4.01, 4.0199, 0.001)
    #        cell._cell.record_times = self.cells[1]._cell.record_times = numpy.arange(0.0, 1.0, 0.1)
    #    simulator.state.t = simulator.state.dt * len(cell._cell.gsyn_trace['excitatory'])
    #    gdata = self.rg._get_current_segment(variables=['gsyn_exc', 'gsyn_inh'], filter_ids=None)
    #    self.assertEqual(len(gdata.analogsignalarrays), 2)
    #    assert_array_equal(numpy.array(gdata.analogsignalarrays[0][:,0]),
    #                        cell._cell.gsyn_trace['excitatory'])
    #    assert_array_equal(numpy.array(gdata.analogsignalarrays[1][:,0]),
    #                        cell._cell.gsyn_trace['inhibitory'])
    #
    def test__local_count(self):
        self.rec.recorded['spikes'] = self.cells
        self.cells[0]._cell.spike_times = h.Vector(numpy.arange(101.0, 111.0))
        self.cells[1]._cell.spike_times = h.Vector(numpy.arange(13.0, 33.0))
        self.assertEqual(self.rec._local_count('spikes', filter_ids=None),
                         {self.cells[0]: 10, self.cells[1]: 20})


@unittest.skipUnless(sim, "Requires NEURON")
class TestStandardIF(unittest.TestCase):
    
    def test_create_cond_exp(self):
        cell = cells.StandardIF("conductance", "exp", tau_m=12.3, c_m=0.246, v_rest=-67.8)
        self.assertAlmostEqual(cell.area(), 1e5, places=10) # µm²
        self.assertEqual(cell(0.5).cm, 0.246)
        self.assertEqual(cell(0.5).pas.g, 2e-5)

    def test_get_attributes(self):
        cell = cells.StandardIF("conductance", "exp", tau_m=12.3, c_m=0.246, v_rest=-67.8)
        self.assertAlmostEqual(cell.tau_m, 12.3, places=10)
        self.assertAlmostEqual(cell.c_m, 0.246, places=10)


if __name__ == '__main__':
    unittest.main()