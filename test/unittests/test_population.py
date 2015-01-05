"""
Tests of the common implementation of the Population class, using the pyNN.mock
backend.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
try:
    basestring
except NameError:
    basestring = str
import numpy
import sys
from numpy.testing import assert_array_equal, assert_array_almost_equal
import quantities as pq
try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch
from .mocks import MockRNG
import pyNN.mock as sim
from pyNN import random, errors, space
from pyNN.parameters import Sequence

from .backends.registry import register_class, register
 
def setUp():
    pass

def tearDown():
    pass
 
@register_class()
class PopulationTest(unittest.TestCase):
    
    def setUp(self, sim=sim, **extra):
        sim.setup(**extra)
        
    def tearDown(self, sim=sim):
        sim.end()
        
    @register()
    def test_create_with_standard_cell_simple(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        self.assertEqual(p.size, 11)
        self.assertIsInstance(p.label, basestring)
        self.assertIsInstance(p.celltype, sim.IF_cond_exp)
        self.assertIsInstance(p._structure, space.Line)
        self.assertEqual(p._positions, None)
        self.assertEqual(p.initial_values.keys(), p.celltype.default_initial_values.keys())

    @register()
    def test_create_with_parameters(self, sim=sim):
        p = sim.Population(4, sim.IF_cond_exp(**{'tau_m': 12.3,
                                                    'tau_syn_E': lambda i: 0.987 + 0.01*i,
                                                    'tau_syn_I': numpy.array([0.5, 0.6, 0.7, 0.8])}))
        tau_syn_E, tau_m, tau_syn_I = p.get(('tau_syn_E', 'tau_m', 'tau_syn_I'), gather=True)
        assert_array_almost_equal(tau_syn_E, numpy.array([0.987, 0.997, 1.007, 1.017]))
        self.assertAlmostEqual(tau_m, 12.3)
        assert_array_equal(tau_syn_I, numpy.array([0.5, 0.6, 0.7, 0.8]))

    # test create native cell

    # test create native cell with params

    # test create with structure
    @register()
    def test_create_with_implicit_grid(self, sim=sim):
        p = sim.Population((11,), sim.IF_cond_exp())
        self.assertEqual(p.size, 11)
        self.assertIsInstance(p.structure, space.Line)
        p = sim.Population((5,6), sim.IF_cond_exp())
        self.assertEqual(p.size, 30)
        self.assertIsInstance(p.structure, space.Grid2D)
        p = sim.Population((2,3,4), sim.IF_cond_exp())
        self.assertEqual(p.size, 24)
        self.assertIsInstance(p.structure, space.Grid3D)
        self.assertRaises(Exception, sim.Population, (2,3,4,5), sim.IF_cond_exp())

    #def test_create_with_initial_values():

    @register()
    def test_id_to_index(self, sim=sim):
        p = sim.Population(11, sim.IF_curr_alpha())
        self.assertEqual(p.id_to_index(p[0]), 0)
        self.assertEqual(p.id_to_index(p[10]), 10)

    @register()
    def test_id_to_index_with_array(self, sim=sim):
        p = sim.Population(11, sim.IF_curr_alpha())
        assert_array_equal(p.id_to_index(p.all_cells[3:9:2]), numpy.arange(3,9,2))

    @register()
    def test_id_to_index_with_populationview(self, sim=sim):
        p = sim.Population(11, sim.IF_curr_alpha())
        view = p[3:7]
        self.assertIsInstance(view, sim.PopulationView)
        assert_array_equal(p.id_to_index(view), numpy.arange(3,7))

    @register()
    def test_id_to_index_with_invalid_id(self, sim=sim):
        p = sim.Population(11, sim.IF_curr_alpha())
        self.assertRaises(ValueError, p.id_to_index, p.last_id+1)
        self.assertRaises(ValueError, p.id_to_index, p.first_id-1)

    @register()
    def test_id_to_index_with_invalid_ids(self, sim=sim):
        p = sim.Population(11, sim.IF_curr_alpha())
        self.assertRaises(ValueError, p.id_to_index, [p.first_id-1] + p.all_cells[0:3].tolist())

    #def test_id_to_local_index():

    # test structure property
    @register()
    def test_set_structure(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        p.positions = numpy.arange(33).reshape(3,11)
        new_struct = space.Grid2D()
        p.structure = new_struct
        self.assertEqual(p.structure, new_struct)
        self.assertEqual(p._positions, None)

    # test positions property
    @register()
    def test_get_positions(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        pos1 = numpy.arange(33).reshape(3,11)
        p._structure = Mock()
        p._structure.generate_positions = Mock(return_value=pos1)
        self.assertEqual(p._positions, None)
        assert_array_equal(p.positions, pos1)

        pos2 = 1+numpy.arange(33).reshape(3,11)
        p.positions = pos2
        assert_array_equal(p.positions, pos2)

    @register()
    def test_set_positions(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        assert p._structure is not None
        new_positions = numpy.random.uniform(size=(3,11))
        p.positions = new_positions
        self.assertEqual(p.structure, None)
        assert_array_equal(p.positions, new_positions)
        new_positions[0,0] = 99.9
        self.assertNotEqual(p.positions[0,0], 99.9)

    @register()
    def test_position_generator(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        assert_array_equal(p.position_generator(0), p.positions[:,0])
        assert_array_equal(p.position_generator(10), p.positions[:,10])
        assert_array_equal(p.position_generator(-1), p.positions[:,10])
        assert_array_equal(p.position_generator(-11), p.positions[:,0])
        self.assertRaises(IndexError, p.position_generator, 11)
        self.assertRaises(IndexError, p.position_generator, -12)

    @register()
    def test__getitem__int(self, sim=sim):
        # Should return the correct ID object
        p = sim.Population(12, sim.IF_cond_exp())
        self.assertEqual(p[11], p[0]+11)
        self.assertRaises(IndexError, p.__getitem__, 12)
        self.assertEqual(p[-1], p[11])

    @register()
    def test__getitem__slice(self, sim=sim):
        # Should return a PopulationView with the correct parent and value
        # of all_cells
        p = sim.Population(17, sim.HH_cond_exp())
        pv = p[3:9]
        self.assertEqual(pv.parent, p)
        assert_array_almost_equal(pv.all_cells, p.all_cells[3:9])

    @register()
    def test__getitem__list(self, sim=sim):
        p = sim.Population(23, sim.HH_cond_exp())
        pv = p[list(range(3,9))]
        self.assertEqual(pv.parent, p)
        assert_array_almost_equal(pv.all_cells, p.all_cells[3:9])

    @register()
    def test__getitem__tuple(self, sim=sim):
        p = sim.Population(23, sim.HH_cond_exp())
        pv = p[(3,5,7)]
        self.assertEqual(pv.parent, p)
        assert_array_almost_equal(pv.all_cells, p.all_cells[[3, 5, 7]])

    @register()
    def test__getitem__invalid(self, sim=sim):
        p = sim.Population(23, sim.IF_curr_alpha())
        self.assertRaises(TypeError, p.__getitem__, "foo")

    @register()
    def test__len__(self, sim=sim):
        # len(p) should give the global size (all MPI nodes)
        p = sim.Population(77, sim.IF_cond_exp())
        self.assertEqual(len(p), p.size, 77)

    @register()
    def test_iter(self, sim=sim):
        p = sim.Population(6, sim.IF_curr_exp())
        itr = p.__iter__()
        assert hasattr(itr, "next") or hasattr(itr, "__next__")
        self.assertEqual(len(list(itr)), 6)

    @register()
    def test___add__two(self, sim=sim):
        # adding two populations should give an Assembly
        p1 = sim.Population(6, sim.IF_curr_exp())
        p2 = sim.Population(17, sim.IF_cond_exp())
        assembly = p1 + p2
        self.assertIsInstance(assembly, sim.Assembly)
        self.assertEqual(assembly.populations, [p1, p2])

    @register()
    def test___add__three(self, sim=sim):
        # adding three populations should give an Assembly
        p1 = sim.Population(6, sim.IF_curr_exp())
        p2 = sim.Population(17, sim.IF_cond_exp())
        p3 = sim.Population(9, sim.HH_cond_exp())
        assembly = p1 + p2 + p3
        self.assertIsInstance(assembly, sim.Assembly)
        self.assertEqual(assembly.populations, [p1, p2, p3])

    @register()
    def test_nearest(self, sim=sim):
        p = sim.Population(13, sim.IF_cond_exp())
        p.positions = numpy.arange(39).reshape((13,3)).T
        self.assertEqual(p.nearest((0.0, 1.0, 2.0)), p[0])
        self.assertEqual(p.nearest((3.0, 4.0, 5.0)), p[1])
        self.assertEqual(p.nearest((36.0, 37.0, 38.0)), p[12])
        self.assertEqual(p.nearest((1.49, 2.49, 3.49)), p[0])
        self.assertEqual(p.nearest((1.51, 2.51, 3.51)), p[1])

        x,y,z = 4,5,6
        p = sim.Population((x,y,z), sim.IF_cond_alpha())
        self.assertEqual(p.nearest((0.0,0.0,0.0)), p[0])
        self.assertEqual(p.nearest((0.0,0.0,1.0)), p[1])
        self.assertEqual(p.nearest((0.0,1.0,0.0)), p[z])
        self.assertEqual(p.nearest((1.0,0.0,0.0)), p[y*z])
        self.assertEqual(p.nearest((3.0,2.0,1.0)), p[3*y*z+2*z+1])
        self.assertEqual(p.nearest((3.49,2.49,1.49)), p[3*y*z+2*z+1])
        self.assertEqual(p.nearest((3.49,2.49,1.51)), p[3*y*z+2*z+2])
        #self.assertEqual(p.nearest((3.49,2.49,1.5)), p[3*y*z+2*z+2]) # known to fail
        #self.assertEqual(p.nearest((2.5,2.5,1.5)), p[3*y*z+3*y+2])

    @register()
    def test_sample(self, sim=sim):
        p = sim.Population(13, sim.IF_cond_exp())
        rng = Mock()
        rng.permutation = Mock(return_value=numpy.array([7,4,8,12,0,3,9,1,2,11,5,10,6]))
        pv = p.sample(5, rng=rng)
        assert_array_equal(pv.all_cells,
                            p.all_cells[[7,4,8,12,0]])

    @register()
    def test_get_multiple_homogeneous_params_with_gather(self, sim=sim):
        p = sim.Population(4, sim.IF_cond_exp(**{'tau_m': 12.3, 'tau_syn_E': 0.987, 'tau_syn_I': 0.7}))
        tau_syn_E, tau_m = p.get(('tau_syn_E', 'tau_m'), gather=True)
        self.assertIsInstance(tau_syn_E, float)
        self.assertEqual(tau_syn_E, 0.987)
        self.assertAlmostEqual(tau_m, 12.3)

    @register()
    def test_get_single_param_with_gather(self, sim=sim):
        p = sim.Population(4, sim.IF_cond_exp(tau_m=12.3, tau_syn_E=0.987, tau_syn_I=0.7))
        tau_syn_E = p.get('tau_syn_E', gather=True)
        self.assertEqual(tau_syn_E, 0.987)

    @register()
    def test_get_multiple_inhomogeneous_params_with_gather(self, sim=sim):
        p = sim.Population(4, sim.IF_cond_exp(tau_m=12.3,
                                                tau_syn_E=[0.987, 0.988, 0.989, 0.990],
                                                tau_syn_I=lambda i: 0.5+0.1*i))
        tau_syn_E, tau_m, tau_syn_I = p.get(('tau_syn_E', 'tau_m', 'tau_syn_I'), gather=True)
        self.assertIsInstance(tau_m, float)
        self.assertIsInstance(tau_syn_E, numpy.ndarray)
        assert_array_equal(tau_syn_E, numpy.array([0.987, 0.988, 0.989, 0.990]))
        self.assertAlmostEqual(tau_m, 12.3)
        assert_array_almost_equal(tau_syn_I, numpy.array([0.5, 0.6, 0.7, 0.8]), decimal=12)

    @register(exclude=['nest', 'neuron', 'brian', 'hardware.brainscales', 'spiNNaker'])
    def test_get_multiple_params_no_gather(self, sim=sim):
        sim.simulator.state.num_processes = 2
        sim.simulator.state.mpi_rank = 1
        p = sim.Population(4, sim.IF_cond_exp(tau_m=12.3,
                                                tau_syn_E=[0.987, 0.988, 0.989, 0.990],
                                                i_offset=lambda i: -0.2*i))
        tau_syn_E, tau_m, i_offset = p.get(('tau_syn_E', 'tau_m', 'i_offset'), gather=False)
        self.assertIsInstance(tau_m, float)
        self.assertIsInstance(tau_syn_E, numpy.ndarray)
        assert_array_equal(tau_syn_E, numpy.array([0.988, 0.990]))
        self.assertEqual(tau_m, 12.3)
        assert_array_almost_equal(i_offset, numpy.array([-0.2, -0.6]), decimal=12)
        sim.simulator.state.num_processes = 1
        sim.simulator.state.mpi_rank = 0

    @register()
    def test_get_sequence_param(self, sim=sim):
        p = sim.Population(3, sim.SpikeSourceArray(spike_times=[Sequence([1, 2, 3, 4]),
                                                                Sequence([2, 3, 4, 5]),
                                                                Sequence([3, 4, 5, 6])]))
        spike_times = p.get('spike_times')
        self.assertEqual(spike_times.size, 3)
        assert_array_equal(spike_times[1], Sequence([2, 3, 4, 5]))

    @register()
    def test_set(self, sim=sim):
        p = sim.Population(4, sim.IF_cond_exp, {'tau_m': 12.3, 'tau_syn_E': 0.987, 'tau_syn_I': 0.7})
        rng = MockRNG(start=1.21, delta=0.01, parallel_safe=True)
        p.set(tau_syn_E=random.RandomDistribution('uniform', (0.5, 1.5), rng=rng), tau_m=9.87)
        tau_m, tau_syn_E, tau_syn_I = p.get(('tau_m', 'tau_syn_E', 'tau_syn_I'), gather=True)
        assert_array_equal(tau_syn_E, numpy.array([1.21, 1.22, 1.23, 1.24]))
        assert_array_almost_equal(tau_m, 9.87*numpy.ones((4,)))
        assert_array_equal(tau_syn_I, 0.7*numpy.ones((4,)))

    @register()
    def test_set_invalid_name(self, sim=sim):
        p = sim.Population(9, sim.HH_cond_exp())
        self.assertRaises(errors.NonExistentParameterError, p.set, foo=13.2)

    @register()
    def test_set_invalid_type(self, sim=sim):
        p = sim.Population(9, sim.IF_cond_exp())
        self.assertRaises(errors.InvalidParameterValueError, p.set, tau_m={})
        self.assertRaises(errors.InvalidParameterValueError, p.set, v_reset='bar')

    @register()
    def test_set_sequence(self, sim=sim):
        p = sim.Population(3, sim.SpikeSourceArray())
        p.set(spike_times=[Sequence([1, 2, 3, 4]),
                            Sequence([2, 3, 4, 5]),
                            Sequence([3, 4, 5, 6])])
        spike_times = p.get('spike_times', gather=True)
        self.assertEqual(spike_times.size, 3)
        assert_array_equal(spike_times[1], Sequence([2, 3, 4, 5]))
        
    @register()
    def test_set_array(self, sim=sim):
        p = sim.Population(5, sim.IF_cond_exp())
        p.set(v_thresh=-50.0+numpy.arange(5))
        assert_array_equal(p.get('v_thresh', gather=True),
                            numpy.array([-50.0, -49.0, -48.0, -47.0, -46.0]))

    @register(exclude=['nest', 'neuron', 'brian', 'hardware.brainscales', 'spiNNaker'])
    def test_set_random_distribution_parallel_unsafe(self, sim=sim):
        orig_rcfg = random.get_mpi_config
        random.get_mpi_config = lambda: (1, 2)
        sim.simulator.state.num_processes = 2
        sim.simulator.state.mpi_rank = 1
        p = sim.Population(4, sim.IF_cond_exp(tau_syn_E=0.987))
        rng = MockRNG(start=1.21, delta=0.01, parallel_safe=False)
        p.set(tau_syn_E=random.RandomDistribution('uniform', (0.8, 1.2), rng=rng))
        tau_syn_E = p.get('tau_syn_E', gather=False)
        assert_array_equal(tau_syn_E, numpy.array([1.21, 1.22]))
        random.get_mpi_config = orig_rcfg
        sim.simulator.state.num_processes = 1
        sim.simulator.state.mpi_rank = 0

    @register(exclude=['nest', 'neuron', 'brian', 'hardware.brainscales', 'spiNNaker'])
    def test_set_random_distribution_parallel_safe(self, sim=sim):
        orig_rcfg = random.get_mpi_config
        random.get_mpi_config = lambda: (1, 2)
        sim.simulator.state.num_processes = 2
        sim.simulator.state.mpi_rank = 1
        p = sim.Population(4, sim.IF_cond_exp(tau_syn_E=0.987))
        rng = MockRNG(start=1.21, delta=0.01, parallel_safe=True)
        p.set(tau_syn_E=random.RandomDistribution('uniform', (0.1, 1), rng=rng))
        tau_syn_E = p.get('tau_syn_E', gather=False)
        assert_array_equal(tau_syn_E, numpy.array([1.22, 1.24]))
        random.get_mpi_config = orig_rcfg
        sim.simulator.state.num_processes = 1
        sim.simulator.state.mpi_rank = 0

    @register()
    def test_tset(self, sim=sim):
        p = sim.Population(17, sim.IF_cond_alpha())
        p.set = Mock()
        tau_m = numpy.linspace(10.0, 20.0, num=p.size)
        p.tset("tau_m", tau_m)
        p.set.assert_called_with(tau_m=tau_m)

    @register()
    def test_rset(self, sim=sim):
        p = sim.Population(17, sim.IF_cond_alpha())
        p.set = Mock()
        v_rest = random.RandomDistribution('uniform', low=-70.0, high=-60.0)
        p.rset("v_rest", v_rest)
        p.set.assert_called_with(v_rest=v_rest)

    ##def test_set_with_native_rng():

    @register()
    def test_initialize(self, sim=sim):
        p = sim.Population(17, sim.EIF_cond_exp_isfa_ista())
        v_init = numpy.linspace(-70.0, -60.0, num=p.size)
        w_init = 0.1
        p.initialize(v=v_init, w=w_init)
        assert_array_equal(p.initial_values['v'].evaluate(simplify=True), v_init)
        assert_array_equal(p.initial_values['w'].evaluate(simplify=True), w_init)
        # should call p.record(('v', 'w')) and check that the recorded data starts with the initial value

    @register(exclude=['hardware.brainscales'])
    def test_can_record(self, sim=sim):
        p = sim.Population(17, sim.EIF_cond_exp_isfa_ista())
        assert p.can_record('v')
        assert p.can_record('w')
        assert p.can_record('gsyn_inh')
        assert p.can_record('spikes')
        assert not p.can_record('foo')

    @register()
    def test_record_with_single_variable(self, sim=sim):
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        p.record('v')
        sim.run(12.3)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignalarrays), 1)
        n_values = int(round(12.3/sim.get_time_step())) + 1
        self.assertEqual(data.analogsignalarrays[0].name, 'v')
        self.assertEqual(data.analogsignalarrays[0].shape, (n_values, p.size))

    @register(exclude=['hardware.brainscales'])
    def test_record_with_multiple_variables(self, sim=sim):
        p = sim.Population(2, sim.EIF_cond_exp_isfa_ista())
        p.record(('v', 'w', 'gsyn_exc'))
        sim.run(10.0)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignalarrays), 3)
        n_values = int(round(10.0/sim.get_time_step())) + 1
        names = set(arr.name for arr in data.analogsignalarrays)
        self.assertEqual(names, set(('v', 'w', 'gsyn_exc')))
        for arr in data.analogsignalarrays:
            self.assertEqual(arr.shape, (n_values, p.size))
            
    @register()
    def test_record_with_v_and_spikes(self, sim=sim):
        p = sim.Population(2, sim.EIF_cond_exp_isfa_ista())
        p.record(('v', 'spikes'))
        sim.run(10.0)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignalarrays), 1)
        n_values = int(round(10.0/sim.get_time_step())) + 1
        names = set(arr.name for arr in data.analogsignalarrays)
        self.assertEqual(names, set(('v')))
        for arr in data.analogsignalarrays:
            self.assertEqual(arr.shape, (n_values, p.size))

    @register()
    def test_record_v(self, sim=sim):
        p = sim.Population(2, sim.EIF_cond_exp_isfa_ista())
        p.record = Mock()
        p.record_v("arg1")
        p.record.assert_called_with('v', "arg1")

    @register()
    def test_record_gsyn(self, sim=sim):
        p = sim.Population(2, sim.EIF_cond_exp_isfa_ista())
        p.record = Mock()
        p.record_gsyn("arg1")
        p.record.assert_called_with(['gsyn_exc', 'gsyn_inh'], "arg1")

    @register()
    def test_record_invalid_variable(self, sim=sim):
        p = sim.Population(14, sim.IF_curr_alpha())
        self.assertRaises(errors.RecordingError,
                            p.record, ('v', 'gsyn_exc')) # can't record gsyn_exc from this celltype

    #def test_write_data(self, sim=sim):
    #    self.fail()
    #

    @register(exclude=['hardware.brainscales'])
    def test_get_data_with_gather(self, sim=sim):
        t1 = 12.3
        t2 = 13.4
        t3 = 14.5
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        p.record('v')
        sim.run(t1)
        # what if we call p.record between two run statements?
        # would be nice to get an AnalogSignalArray with a non-zero t_start
        # but then need to make sure we get the right initial value
        sim.run(t2)
        sim.reset()
        p.record('spikes')
        p.record('w')
        sim.run(t3)
        data = p.get_data(gather=True)
        self.assertEqual(len(data.segments), 2)

        seg0 = data.segments[0]
        self.assertEqual(len(seg0.analogsignalarrays), 1)
        v = seg0.analogsignalarrays[0]
        self.assertEqual(v.name, 'v')
        num_points = int(round((t1 + t2)/sim.get_time_step())) + 1
        self.assertEqual(v.shape, (num_points, p.size))
        self.assertEqual(v.t_start, 0.0*pq.ms)
        self.assertEqual(v.units, pq.mV)
        self.assertEqual(v.sampling_period, 0.1*pq.ms)
        self.assertEqual(len(seg0.spiketrains), 0)

        seg1 = data.segments[1]
        self.assertEqual(len(seg1.analogsignalarrays), 2)
        w = seg1.filter(name='w')[0]
        self.assertEqual(w.name, 'w')
        num_points = int(round(t3/sim.get_time_step())) + 1
        self.assertEqual(w.shape, (num_points, p.size))
        self.assertEqual(v.t_start, 0.0)
        self.assertEqual(len(seg1.spiketrains), p.size)
            
    @register(exclude=['nest', 'neuron', 'brian', 'hardware.brainscales', 'spiNNaker'])
    def test_get_spikes_with_gather(self, sim=sim):
        t1 = 12.3
        t2 = 13.4
        t3 = 14.5
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        p.record('v')
        sim.run(t1)
        sim.run(t2)
        sim.reset()
        p.record('spikes')
        p.record('w')
        sim.run(t3)
        data = p.get_data(gather=True)
        self.assertEqual(len(data.segments), 2)

        seg0 = data.segments[0]
        self.assertEqual(len(seg0.analogsignalarrays), 1)
        self.assertEqual(len(seg0.spiketrains), 0)

        seg1 = data.segments[1]
        self.assertEqual(len(seg1.analogsignalarrays), 2)
        self.assertEqual(len(seg1.spiketrains), p.size)
        assert_array_equal(seg1.spiketrains[7],
                            numpy.array([p.first_id+7, p.first_id+7+5]) % t3)

    #def test_get_data_no_gather(self, sim=sim):
    #    self.fail()

    @register()
    def test_printSpikes(self, sim=sim):
        # TODO: implement assert_deprecated
        p = sim.Population(3, sim.IF_curr_alpha())
        p.record('spikes')
        sim.run(10.0)
        p.write_data = Mock()
        p.printSpikes("foo.txt")
        p.write_data.assert_called_with('foo.txt', 'spikes', True)

    @register()
    def test_getSpikes(self, sim=sim):
        p = sim.Population(3, sim.IF_curr_alpha())
        p.record('spikes')
        sim.run(10.0)
        p.get_data = Mock()
        p.getSpikes()
        p.get_data.assert_called_with('spikes', True)

    @register()
    def test_print_v(self, sim=sim):
        p = sim.Population(3, sim.IF_curr_alpha())
        p.record_v()
        sim.run(10.0)
        p.write_data = Mock()
        p.print_v("foo.txt")
        p.write_data.assert_called_with('foo.txt', 'v', True)

    @register()
    def test_get_v(self, sim=sim):
        p = sim.Population(3, sim.IF_curr_alpha())
        p.record_v()
        sim.run(10.0)
        p.get_data = Mock()
        p.get_v()
        p.get_data.assert_called_with('v', True)

    @register(exclude=['hardware.brainscales'])
    def test_print_gsyn(self, sim=sim):
        p = sim.Population(3, sim.IF_cond_alpha())
        p.record_gsyn()
        sim.run(10.0)
        p.write_data = Mock()
        p.print_gsyn("foo.txt")
        p.write_data.assert_called_with('foo.txt', ['gsyn_exc', 'gsyn_inh'], True)

    @register(exclude=['hardware.brainscales'])
    def test_get_gsyn(self, sim=sim):
        p = sim.Population(3, sim.IF_cond_alpha())
        p.record_gsyn()
        sim.run(10.0)
        p.get_data = Mock()
        p.get_gsyn()
        p.get_data.assert_called_with(['gsyn_exc', 'gsyn_inh'], True)

    @register(exclude=['nest', 'neuron', 'brian', 'hardware.brainscales', 'spiNNaker'])
    def test_get_spike_counts(self, sim=sim):
        p = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        p.record('spikes')
        sim.run(100.0)
        self.assertEqual(p.get_spike_counts(),
                            {p.all_cells[0]: 2,
                            p.all_cells[1]: 2,
                            p.all_cells[2]: 2})

    @register(exclude=['nest', 'neuron', 'brian', 'hardware.brainscales', 'spiNNaker'])
    def test_mean_spike_count(self, sim=sim):
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        p.record('spikes')
        sim.run(100.0)
        self.assertEqual(p.mean_spike_count(), 2.0)

    ##def test_mean_spike_count_on_slave_node():

    @register()
    def test_meanSpikeCount(self, sim=sim):
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        p.record('spikes')
        sim.run(100.0)
        p.mean_spike_count = Mock()
        p.meanSpikeCount()
        p.mean_spike_count.assert_called()

    @register()
    def test_inject(self, sim=sim):
        p = sim.Population(3, sim.IF_curr_alpha())
        cs = Mock()
        p.inject(cs)
        meth, args, kwargs = cs.method_calls[0]
        self.assertEqual(meth, "inject_into")
        self.assertEqual(args, (p,))

    @register()
    def test_inject_into_invalid_celltype(self, sim=sim):
        p = sim.Population(3, sim.SpikeSourceArray())
        self.assertRaises(TypeError, p.inject, Mock())

    #def test_save_positions(self, sim=sim):
    #    self.fail()

    # test describe method
    @register()
    def test_describe(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        self.assertIsInstance(p.describe(), basestring)
        self.assertIsInstance(p.describe(template=None), dict)

if __name__ == "__main__":
    unittest.main()
