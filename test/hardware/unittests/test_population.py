"""
Tests of the common implementation of the Population class, using the pyNN.mock
backend.

:copyright: Copyright 2006-2019 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal
import quantities as pq
from mock import Mock
from mocks import MockRNG
from pyNN import random, errors, space
from pyNN.parameters import Sequence

if True:
    import pyNN.hardware.brainscales as sim
    ParameterValueOutOfRangeError = sim.range_checker.ParameterValueOutOfRangeError
    IF_cond_exp = sim.Hardware_IF_cond_exp
    EIF_cond_exp_isfa_ista = sim.Hardware_EIF_cond_exp_isfa_ista
else:
    import pyNN.mock as sim
    IF_cond_exp = sim.IF_cond_exp
    EIF_cond_exp_isfa_ista = sim.EIF_cond_exp_isfa_ista


class PopulationTest(unittest.TestCase):

    def setUp(self):
        if True:
            extra = {'loglevel': 0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        else:
            extra = {}
        sim.setup(**extra)
        
    def tearDown(self):
        sim.end()

    def test_create_with_standard_cell_simple(self):

        p = sim.Population(11, IF_cond_exp())
        self.assertEqual(p.size, 11)
        self.assertIsInstance(p.label, basestring)
        self.assertIsInstance(p.celltype, IF_cond_exp)
        self.assertIsInstance(p._structure, space.Line)
        self.assertEqual(p._positions, None)
        self.assertEqual(p.initial_values.keys(), p.celltype.default_initial_values.keys())

    def test_create_with_parameters(self):
        p = sim.Population(4, IF_cond_exp(**{'tau_m': lambda i: 12.3 + 0.1 * i,
                                                 'cm': 0.2,
                                                 'tau_syn_E': numpy.array([1.0, 2.0, 3.0, 4.0])}))
        cm, tau_m, tau_syn_E = p.get(('cm', 'tau_m', 'tau_syn_E'), gather=True)
        assert_array_almost_equal(tau_m, numpy.array([12.3, 12.4, 12.5, 12.6]), decimal=12)
        self.assertEqual(cm, 0.2)
        assert_array_equal(tau_syn_E, numpy.array([1.0, 2.0, 3.0, 4.0]))

    # test create native cell

    # test create native cell with params

    # test create with structure
    def test_create_with_implicit_grid(self):
        p = sim.Population((11,), IF_cond_exp())
        self.assertEqual(p.size, 11)
        self.assertIsInstance(p.structure, space.Line)
        p = sim.Population((5, 6), IF_cond_exp())
        self.assertEqual(p.size, 30)
        self.assertIsInstance(p.structure, space.Grid2D)
        p = sim.Population((2, 3, 4), IF_cond_exp())
        self.assertEqual(p.size, 24)
        self.assertIsInstance(p.structure, space.Grid3D)
        self.assertRaises(Exception, sim.Population, (2, 3, 4, 5), IF_cond_exp())

    #def test_create_with_initial_values():

    def test_id_to_index(self):
        p = sim.Population(11, IF_cond_exp())
        self.assertEqual(p.id_to_index(p[0]), 0)
        self.assertEqual(p.id_to_index(p[10]), 10)

    def test_id_to_index_with_array(self):
        p = sim.Population(11, IF_cond_exp())
        assert_array_equal(p.id_to_index(p.all_cells[3:9:2]), numpy.arange(3, 9, 2))

    def test_id_to_index_with_populationview(self):
        p = sim.Population(11, IF_cond_exp())
        view = p[3:7]
        self.assertIsInstance(view, sim.PopulationView)
        assert_array_equal(p.id_to_index(view), numpy.arange(3, 7))

    def test_id_to_index_with_invalid_id(self):
        p = sim.Population(11, IF_cond_exp())
        self.assertRaises(ValueError, p.id_to_index, p.last_id + 1)
        self.assertRaises(ValueError, p.id_to_index, p.first_id - 1)

    def test_id_to_index_with_invalid_ids(self):
        p = sim.Population(11, IF_cond_exp())
        self.assertRaises(ValueError, p.id_to_index, [p.first_id - 1] + p.all_cells[0:3].tolist())

    #def test_id_to_local_index():

    # test structure property
    def test_set_structure(self):
        p = sim.Population(11, IF_cond_exp())
        p.positions = numpy.arange(33).reshape(3, 11)
        new_struct = space.Grid2D()
        p.structure = new_struct
        self.assertEqual(p.structure, new_struct)
        self.assertEqual(p._positions, None)

    # test positions property
    def test_get_positions(self):
        p = sim.Population(11, IF_cond_exp())
        pos1 = numpy.arange(33).reshape(3, 11)
        p._structure = Mock()
        p._structure.generate_positions = Mock(return_value=pos1)
        self.assertEqual(p._positions, None)
        assert_array_equal(p.positions, pos1)

        pos2 = 1 + numpy.arange(33).reshape(3, 11)
        p.positions = pos2
        assert_array_equal(p.positions, pos2)

    def test_set_positions(self):
        p = sim.Population(11, IF_cond_exp())
        assert p._structure != None
        new_positions = numpy.random.uniform(size=(3, 11))
        p.positions = new_positions
        self.assertEqual(p.structure, None)
        assert_array_equal(p.positions, new_positions)
        new_positions[0, 0] = 99.9
        self.assertNotEqual(p.positions[0, 0], 99.9)

    def test_position_generator(self):
        p = sim.Population(11, IF_cond_exp())
        assert_array_equal(p.position_generator(0), p.positions[:, 0])
        assert_array_equal(p.position_generator(10), p.positions[:, 10])
        assert_array_equal(p.position_generator(-1), p.positions[:, 10])
        assert_array_equal(p.position_generator(-11), p.positions[:, 0])
        self.assertRaises(IndexError, p.position_generator, 11)
        self.assertRaises(IndexError, p.position_generator, -12)

    def test__getitem__int(self):
        # Should return the correct ID object
        p = sim.Population(12, IF_cond_exp())
        self.assertEqual(p[11], 11 + p[0])
        self.assertRaises(IndexError, p.__getitem__, 12)
        self.assertEqual(p[-1], 11 + p[0])

    def test__getitem__slice(self):
        # Should return a PopulationView with the correct parent and value
        # of all_cells
        p = sim.Population(17, IF_cond_exp())
        pv = p[3:9]
        self.assertEqual(pv.parent, p)
        assert_array_almost_equal(pv.all_cells, p.all_cells[3:9])

    def test__getitem__list(self):
       p = sim.Population(23, IF_cond_exp())
       pv = p[range(3, 9)]
       self.assertEqual(pv.parent, p)
       assert_array_almost_equal(pv.all_cells, p.all_cells[3:9])

    def test__getitem__tuple(self):
        p = sim.Population(23, IF_cond_exp())
        pv = p[(3, 5, 7)]
        self.assertEqual(pv.parent, p)
        assert_array_almost_equal(pv.all_cells, p.all_cells[[3, 5, 7]])

    def test__getitem__invalid(self):
        p = sim.Population(23, IF_cond_exp())
        self.assertRaises(TypeError, p.__getitem__, "foo")

    def test__len__(self):
        # len(p) should give the global size (all MPI nodes)
        p = sim.Population(77, IF_cond_exp())
        self.assertEqual(len(p), p.size, 77)

    def test_iter(self):
        p = sim.Population(6, IF_cond_exp())
        itr = p.__iter__()
        assert hasattr(itr, "next")
        self.assertEqual(len(list(itr)), 6)

    def test___add__two(self):
        # adding two populations should give an Assembly
        p1 = sim.Population(6, EIF_cond_exp_isfa_ista())
        p2 = sim.Population(17, IF_cond_exp())
        assembly = p1 + p2
        self.assertIsInstance(assembly, sim.Assembly)
        self.assertEqual(assembly.populations, [p1, p2])

    def test___add__three(self):
        # adding three populations should give an Assembly
        p1 = sim.Population(6, IF_cond_exp())
        p2 = sim.Population(17, EIF_cond_exp_isfa_ista())
        p3 = sim.Population(9, sim.SpikeSourceArray())
        assembly = p1 + p2 + p3
        self.assertIsInstance(assembly, sim.Assembly)
        self.assertEqual(assembly.populations, [p1, p2, p3])

    def test_nearest(self):
        p = sim.Population(13, IF_cond_exp())
        p.positions = numpy.arange(39).reshape((13, 3)).T
        self.assertEqual(p.nearest((0.0, 1.0, 2.0)), p[0])
        self.assertEqual(p.nearest((3.0, 4.0, 5.0)), p[1])
        self.assertEqual(p.nearest((36.0, 37.0, 38.0)), p[12])
        self.assertEqual(p.nearest((1.49, 2.49, 3.49)), p[0])
        self.assertEqual(p.nearest((1.51, 2.51, 3.51)), p[1])

        x, y, z = 4, 5, 6
        p = sim.Population((x, y, z), IF_cond_exp())
        self.assertEqual(p.nearest((0.0, 0.0, 0.0)), p[0])
        self.assertEqual(p.nearest((0.0, 0.0, 1.0)), p[1])
        self.assertEqual(p.nearest((0.0, 1.0, 0.0)), p[z])
        self.assertEqual(p.nearest((1.0, 0.0, 0.0)), p[y * z])
        self.assertEqual(p.nearest((3.0, 2.0, 1.0)), p[3 * y * z + 2 * z + 1])
        self.assertEqual(p.nearest((3.49, 2.49, 1.49)), p[3 * y * z + 2 * z + 1])
        self.assertEqual(p.nearest((3.49, 2.49, 1.51)), p[3 * y * z + 2 * z + 2])
        #self.assertEqual(p.nearest((3.49,2.49,1.5)), p[3*y*z+2*z+2]) # known to fail
        #self.assertEqual(p.nearest((2.5,2.5,1.5)), p[3*y*z+3*y+2])

    def test_sample(self):
        p = sim.Population(13, IF_cond_exp())
        rng = Mock()
        rng.permutation = Mock(return_value=numpy.array([7, 4, 8, 12, 0, 3, 9, 1, 2, 11, 5, 10, 6]))
        pv = p.sample(5, rng=rng)
        assert_array_equal(pv.all_cells,
                           p.all_cells[[7, 4, 8, 12, 0]])

    def test_get_multiple_homogeneous_params_with_gather(self):
        p = sim.Population(4, IF_cond_exp(**{'tau_m': 12.3, 'cm': 0.2, 'i_offset': 0.0}))
        cm, tau_m = p.get(('cm', 'tau_m'), gather=True)
        self.assertIsInstance(cm, float)
        self.assertEqual(cm, 0.2)
        self.assertEqual(tau_m, 12.3)

    def test_get_single_param_with_gather(self):
        p = sim.Population(4, IF_cond_exp(tau_m=12.3, cm=0.2, i_offset=0.))
        tau_m = p.get('tau_m', gather=True)
        self.assertEqual(tau_m, 12.3)

    def test_get_multiple_inhomogeneous_params_with_gather(self):
        p = sim.Population(4, IF_cond_exp(tau_m=[12.3, 12.4, 12.5, 12.6],
                                              cm=0.2,
                                              tau_syn_E=lambda i: 1.0 + 1.0 * i))
        cm, tau_m, tau_syn_E = p.get(('cm', 'tau_m', 'tau_syn_E'), gather=True)
        self.assertIsInstance(cm, float)
        self.assertIsInstance(tau_m, numpy.ndarray)
        assert_array_equal(tau_m, numpy.array([12.3, 12.4, 12.5, 12.6]))
        self.assertEqual(cm, 0.2)
        assert_array_almost_equal(tau_syn_E, numpy.array([1.0, 2.0, 3.0, 4.0]), decimal=12)

    def test_get_sequence_param(self):
        p = sim.Population(3, sim.SpikeSourceArray(spike_times=[Sequence([1, 2, 3, 4]),
                                                                Sequence([2, 3, 4, 5]),
                                                                Sequence([3, 4, 5, 6])]))
        spike_times = p.get('spike_times')
        self.assertEqual(spike_times.size, 3)
        assert_array_equal(spike_times[1], Sequence([2, 3, 4, 5]))

    def test_set(self):
        p = sim.Population(4, IF_cond_exp, {'tau_m': 14.0, 'v_reset': -50., 'tau_syn_E': 1.0})
        rng = MockRNG(start=12.31, delta=0.01, parallel_safe=True)
        p.set(tau_m=random.RandomDistribution('uniform', (0.5, 1.5), rng=rng), v_reset=-60)
        tau_m, v_reset, tau_syn_E = p.get(('tau_m', 'v_reset', 'tau_syn_E'), gather=True)
        assert_array_equal(tau_m, numpy.array([12.31, 12.32, 12.33, 12.34]))
        assert_array_equal(v_reset, -60.0**numpy.ones((4,)))
        assert_array_equal(tau_syn_E, 1.0 * numpy.ones((4,)))

    def test_set_invalid_name(self):
        p = sim.Population(9, IF_cond_exp())
        self.assertRaises(errors.NonExistentParameterError, p.set, foo=13.2)

    def test_set_invalid_type(self):
        p = sim.Population(9, IF_cond_exp())
        self.assertRaises(errors.InvalidParameterValueError, p.set, tau_m={})
        self.assertRaises(errors.InvalidParameterValueError, p.set, v_reset='bar')

    def test_set_sequence(self):
        p = sim.Population(3, sim.SpikeSourceArray())
        p.set(spike_times=[Sequence([1, 2, 3, 4]),
                           Sequence([2, 3, 4, 5]),
                           Sequence([3, 4, 5, 6])])
        spike_times = p.get('spike_times', gather=True)
        self.assertEqual(spike_times.size, 3)
        assert_array_equal(spike_times[1], Sequence([2, 3, 4, 5]))

    def test_set_array(self):
        p = sim.Population(5, IF_cond_exp())
        p.set(v_thresh=-50.0 + numpy.arange(5))
        assert_array_equal(p.get('v_thresh', gather=True),
                           numpy.array([-50.0, -49.0, -48.0, -47.0, -46.0]))

    def test_set_random_distribution_parallel_unsafe(self):
        orig_rcfg = random.get_mpi_config
        random.get_mpi_config = lambda: (1, 2)
        sim.simulator.state.num_processes = 2
        sim.simulator.state.mpi_rank = 1
        p = sim.Population(4, IF_cond_exp(tau_syn_E=1.0))
        rng = MockRNG(start=1.5, delta=0.01, parallel_safe=False)
        p.set(tau_syn_E=random.RandomDistribution('uniform', (0.8, 3.0), rng=rng))
        tau_syn_E = p.get('tau_syn_E', gather=False)
        assert_array_equal(tau_syn_E, numpy.array([1.5, 1.51]))
        random.get_mpi_config = orig_rcfg
        sim.simulator.state.num_processes = 1
        sim.simulator.state.mpi_rank = 0

    def test_set_random_distribution_parallel_safe(self):
        orig_rcfg = random.get_mpi_config
        random.get_mpi_config = lambda: (1, 2)
        sim.simulator.state.num_processes = 2
        sim.simulator.state.mpi_rank = 1
        p = sim.Population(4, IF_cond_exp(tau_syn_E=1.0))
        rng = MockRNG(start=2.0, delta=0.01, parallel_safe=True)
        p.set(tau_syn_E=random.RandomDistribution('uniform', (0.1, 3.0), rng=rng))
        tau_syn_E = p.get('tau_syn_E', gather=False)
        assert_array_equal(tau_syn_E, numpy.array([2.01, 2.03]))
        random.get_mpi_config = orig_rcfg
        sim.simulator.state.num_processes = 1
        sim.simulator.state.mpi_rank = 0

    def test_tset(self):
        p = sim.Population(17, IF_cond_exp())
        p.set = Mock()
        tau_m = numpy.linspace(10.0, 20.0, num=p.size)
        p.tset("tau_m", tau_m)
        p.set.assert_called_with(tau_m=tau_m)

    def test_rset(self):
        p = sim.Population(17, IF_cond_exp())
        p.set = Mock()
        v_reset = random.RandomDistribution('uniform', low=-70.0, high=-60.0)
        p.rset("v_reset", v_reset)
        p.set.assert_called_with(v_reset=v_reset)

    ##def test_set_with_native_rng():

    def test_initialize(self):
        p = sim.Population(17, EIF_cond_exp_isfa_ista())
        v_init = numpy.linspace(-70.0, -60.0, num=p.size)
        w_init = 0.1
        p.initialize(v=v_init, w=w_init)
        assert_array_equal(p.initial_values['v'].evaluate(simplify=True), v_init)
        assert_array_equal(p.initial_values['w'].evaluate(simplify=True), w_init)
        # should call p.record(('v', 'w')) and check that the recorded data starts with the initial value

    def test_can_record(self):
        p = sim.Population(17, EIF_cond_exp_isfa_ista())
        assert p.can_record('v')
        assert p.can_record('spikes')
        assert not p.can_record('foo')
        assert not p.can_record('w')
        assert not p.can_record('gsyn_inh')

    def test_record_with_single_variable(self):
        p = sim.Population(14, EIF_cond_exp_isfa_ista())
        p.record('v')
        sim.run(12.3)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignals), 1)
        n_values = int(round(12.3 / sim.get_time_step())) + 1
        self.assertEqual(data.analogsignals[0].name, 'v')
        self.assertEqual(data.analogsignals[0].shape, (n_values, p.size))

    def test_record_with_multiple_variables(self):
        p = sim.Population(2, EIF_cond_exp_isfa_ista())
        p.record(('v', 'spikes'))
        sim.run(10.0)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignals), 1)
        n_values = int(round(10.0 / sim.get_time_step())) + 1
        names = set(arr.name for arr in data.analogsignals)
        self.assertEqual(names, set(('v')))
        for arr in data.analogsignals:
            self.assertEqual(arr.shape, (n_values, p.size))

    def test_record_v(self):
        p = sim.Population(2, EIF_cond_exp_isfa_ista())
        p.record = Mock()
        p.record_v("arg1")
        p.record.assert_called_with('v', "arg1")

    def test_record_gsyn(self):
        p = sim.Population(2, EIF_cond_exp_isfa_ista())
        p.record = Mock()
        p.record_gsyn("arg1")
        p.record.assert_called_with(['gsyn_exc', 'gsyn_inh'], "arg1")

    def test_record_invalid_variable(self):
        p = sim.Population(14, IF_cond_exp())
        self.assertRaises(errors.RecordingError,
                          p.record, ('v', 'w'))  # can't record w from this celltype

    def test_get_data_no_gather(self):
       self.fail()

    def test_printSpikes(self):
        # TODO: implement assert_deprecated
        p = sim.Population(3, IF_cond_exp())
        p.record('spikes')
        sim.run(10.0)
        p.write_data = Mock()
        p.printSpikes("foo.txt")
        p.write_data.assert_called_with('foo.txt', 'spikes', True)

    def test_getSpikes(self):
        p = sim.Population(3, IF_cond_exp())
        p.record('spikes')
        sim.run(10.0)
        p.get_data = Mock()
        p.getSpikes()
        p.get_data.assert_called_with('spikes', True)

    def test_print_v(self):
        p = sim.Population(3, IF_cond_exp())
        p.record_v()
        sim.run(10.0)
        p.write_data = Mock()
        p.print_v("foo.txt")
        p.write_data.assert_called_with('foo.txt', 'v', True)

    def test_get_v(self):
        p = sim.Population(3, IF_cond_exp())
        p.record_v()
        sim.run(10.0)
        p.get_data = Mock()
        p.get_v()
        p.get_data.assert_called_with('v', True)

    def test_get_spike_counts(self):
        p = sim.Population(3, EIF_cond_exp_isfa_ista())
        p.record('spikes')
        sim.run(100.0)
        self.assertEqual(p.get_spike_counts(),
                         {p.all_cells[0]: 2,
                          p.all_cells[1]: 2,
                          p.all_cells[2]: 2})

    def test_mean_spike_count(self):
        p = sim.Population(14, EIF_cond_exp_isfa_ista())
        p.record('spikes')
        sim.run(100.0)
        self.assertEqual(p.mean_spike_count(), 2.0)

    def test_meanSpikeCount(self):
        p = sim.Population(14, EIF_cond_exp_isfa_ista())
        p.record('spikes')
        sim.run(100.0)
        p.mean_spike_count = Mock()
        p.meanSpikeCount()
        self.assertTrue(p.mean_spike_count.called)

    def test_inject(self):
        p = sim.Population(3, IF_cond_exp())
        cs = Mock()
        p.inject(cs)
        meth, args, kwargs = cs.method_calls[0]
        self.assertEqual(meth, "inject_into")
        self.assertEqual(args, (p,))

    def test_inject_into_invalid_celltype(self):
        p = sim.Population(3, sim.SpikeSourceArray())
        self.assertRaises(TypeError, p.inject, Mock())

    # test describe method
    def test_describe(self):
        p = sim.Population(11, IF_cond_exp())
        self.assertIsInstance(p.describe(), basestring)
        self.assertIsInstance(p.describe(template=None), dict)
        
# ------------------------------------------
# specific hardware test
# ------------------------------------------
    #def test_out_of_range_parameter_homogeneous(self):
        #ifcell = IF_cond_exp(cm=0.2)
        #self.assertRaises(ParameterValueOutOfRangeError, IF_cond_exp, cm=0.9)
        
    #def test_out_of_range_parameter_function(self):
        #ifcell = IF_cond_exp(**{'cm': lambda i: 0.2 + 0.1*i})
        #self.assertRaises(ParameterValueOutOfRangeError, sim.Population, 4, ifcell)     
    
    #def test_out_of_range_parameter_array_one_cell(self):
        #self.assertRaises(ParameterValueOutOfRangeError, IF_cond_exp,**{'cm': numpy.array([0.9])})
        
    #def test_out_of_range_parameter_function_one_cell(self):
        #ifcell = IF_cond_exp(**{'cm': lambda i: 0.3 + 0.1*i})
        #self.assertRaises(ParameterValueOutOfRangeError, sim.Population, 1, ifcell)
        
    #def test_out_of_range_parameter_homogeneous_pop(self):
        #self.assertRaises(ParameterValueOutOfRangeError, sim.Population, 4, IF_cond_exp(**{'cm': 0.9}))
        
    #def test_out_of_range_parameter_array_pop(self):
        #self.assertRaises(ParameterValueOutOfRangeError, sim.Population, 4, IF_cond_exp(**{'cm': numpy.array([0.2, 0.2, 0.9, 0.2])}))
        
    #def test_out_of_range_parameter_function_pop(self):
        #p = sim.Population(4, IF_cond_exp(**{'cm': lambda i: 0.2 + 0.0*i}))
        #self.assertRaises(ParameterValueOutOfRangeError, sim.Population, 4, IF_cond_exp(**{'cm': lambda i: 0.2 + 0.1*i}))

if __name__ == "__main__":
    unittest.main()
