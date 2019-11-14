"""
Tests of the common implementation of the PopulationView class, using the
pyNN.mock backend.

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
from mock import Mock, patch
from .mocks import MockRNG
import pyNN.hardware.brainscales as sim
from pyNN import random, errors, space
from pyNN.parameters import Sequence


class PopulationViewTest(unittest.TestCase):

    def setUp(self):
        extra = {'loglevel': 0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        sim.setup(**extra)
        
    def tearDown(self):
        sim.end()

    # test create with population parent and mask selector
    def test_create_with_slice_selector(self):
        p = sim.Population(11, sim.IF_cond_exp())
        mask = slice(3, 9, 2)
        pv = sim.PopulationView(parent=p, selector=mask)
        self.assertEqual(pv.parent, p)
        self.assertEqual(pv.size, 3)
        self.assertEqual(pv.mask, mask)
        assert_array_equal(pv.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
        #assert_array_equal(pv.local_cells, numpy.array([p.all_cells[3]]))
        #assert_array_equal(pv._mask_local, numpy.array([1,0,0], dtype=bool))
        self.assertEqual(pv.celltype, p.celltype)
        self.assertEqual(pv.first_id, p.all_cells[3])
        self.assertEqual(pv.last_id, p.all_cells[7])

    def test_create_with_boolean_array_selector(self):
        p = sim.Population(11, sim.IF_cond_exp())
        mask = numpy.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        pv = sim.PopulationView(parent=p, selector=mask)
        assert_array_equal(pv.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
        #assert_array_equal(pv.mask, mask)

    def test_create_with_index_array_selector(self):
        p = sim.Population(11, sim.EIF_cond_exp_isfa_ista())
        mask = numpy.array([3, 5, 7])
        pv = sim.PopulationView(parent=p, selector=mask)
        assert_array_equal(pv.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
        assert_array_equal(pv.mask, mask)

    # test create with populationview parent and mask selector
    def test_create_with_slice_selector(self):
        p = sim.Population(11, sim.EIF_cond_exp_isfa_ista())
        mask1 = slice(0, 9, 1)
        pv1 = sim.PopulationView(parent=p, selector=mask1)
        assert_array_equal(pv1.all_cells, p.all_cells[0:9])
        mask2 = slice(3, 9, 2)
        pv2 = sim.PopulationView(parent=pv1, selector=mask2)
        self.assertEqual(pv2.parent, pv1)  # or would it be better to resolve the parent chain up to an actual Population?
        assert_array_equal(pv2.all_cells, numpy.array([p.all_cells[3], p.all_cells[5], p.all_cells[7]]))
        #assert_array_equal(pv2._mask_local, numpy.array([1,0,0], dtype=bool))

    # test initial values property

    def test_structure_property(self):
        p = sim.Population(11, sim.SpikeSourcePoisson())
        mask = slice(3, 9, 2)
        pv = sim.PopulationView(parent=p, selector=mask)
        self.assertEqual(pv.structure, p.structure)

    # test positions property
    def test_get_positions(self):
        p = sim.Population(11, sim.IF_cond_exp())
        ppos = numpy.random.uniform(size=(3, 11))
        p._positions = ppos
        pv = sim.PopulationView(parent=p, selector=slice(3, 9, 2))
        assert_array_equal(pv.positions, numpy.array([ppos[:, 3], ppos[:, 5], ppos[:, 7]]).T)

    def test_id_to_index(self):
        p = sim.Population(11, sim.EIF_cond_exp_isfa_ista())
        pv = p[2, 5, 7, 8]
        self.assertEqual(pv.id_to_index(pv[0]), 0)
        self.assertEqual(pv.id_to_index(pv[3]), 3)
        self.assertEqual(pv.id_to_index(p[2]), 0)
        self.assertEqual(pv.id_to_index(p[8]), 3)

    def test_id_to_index_with_array(self):
        p = sim.Population(121, sim.EIF_cond_exp_isfa_ista())
        pv = p[2, 5, 7, 8, 19, 37, 49, 82, 83, 99]
        assert_array_equal(pv.id_to_index(pv.all_cells[3:9:2]), numpy.arange(3, 9, 2))

    def test_id_to_index_with_invalid_id(self):
        p = sim.Population(11, sim.EIF_cond_exp_isfa_ista())
        pv = p[2, 5, 7, 8]
        self.assertRaises(IndexError, pv.id_to_index, p[0])
        self.assertRaises(IndexError, pv.id_to_index, p[9])

#    def test_id_to_index_with_invalid_ids(self):
#        p = sim.Population(11, sim.EIF_cond_exp_isfa_ista())
#        pv = p[2, 5, 7, 8]
#        self.assertRaises(IndexError, pv.id_to_index, p.all_cells[[2, 5, 6]])
# currently failing

    ##def test_id_to_local_index():

    ## test structure property
    def test_set_structure(self):
        p = sim.Population(11, sim.IF_cond_exp(), structure=space.Grid2D())
        pv = p[2, 5, 7, 8]
        new_struct = space.Line()

        def set_struct(struct):
            pv.structure = struct
        self.assertRaises(AttributeError, set_struct, new_struct)

    ## test positions property
    def test_get_positions(self):
        p = sim.Population(11, sim.IF_cond_exp())
        pos = numpy.arange(33).reshape(3, 11)
        p.positions = pos
        pv = p[2, 5, 7, 8]
        assert_array_equal(pv.positions, pos[:, [2, 5, 7, 8]])

    def test_position_generator(self):
        p = sim.Population(11, sim.IF_cond_exp())
        pv = p[2, 5, 7, 8]
        assert_array_equal(pv.position_generator(0), p.positions[:, 2])
        assert_array_equal(pv.position_generator(3), p.positions[:, 8])
        assert_array_equal(pv.position_generator(-1), p.positions[:, 8])
        assert_array_equal(pv.position_generator(-4), p.positions[:, 2])
        self.assertRaises(IndexError, pv.position_generator, 4)
        self.assertRaises(IndexError, pv.position_generator, -5)

    def test__getitem__int(self):
        # Should return the correct ID object
        p = sim.Population(12, sim.IF_cond_exp())
        pv = p[1, 5, 6, 8, 11]

        self.assertEqual(pv[0], p[1], 42)
        self.assertEqual(pv[4], p[11], 53)
        self.assertRaises(IndexError, pv.__getitem__, 6)
        self.assertEqual(pv[-1], p[11], 53)

    def test__getitem__slice(self):
        # Should return a PopulationView with the correct parent and value
        # of all_cells
        p = sim.Population(17, sim.EIF_cond_exp_isfa_ista())
        pv1 = p[1, 5, 6, 8, 11, 12, 15, 16]

        pv2 = pv1[2:6]
        self.assertEqual(pv2.parent, pv1)
        self.assertEqual(pv2.grandparent, p)
        assert_array_equal(pv2.all_cells, pv1.all_cells[[2, 3, 4, 5]])
        assert_array_equal(pv2.all_cells, p.all_cells[[6, 8, 11, 12]])

    def test__getitem__list(self):
       p = sim.Population(23, sim.EIF_cond_exp_isfa_ista())
       pv1 = p[1, 5, 6, 8, 11, 12, 15, 16, 19, 20]

       pv2 = pv1[range(3, 8)]
       self.assertEqual(pv2.parent, pv1)
       assert_array_almost_equal(pv2.all_cells, p.all_cells[[8, 11, 12, 15, 16]])

    def test__getitem__tuple(self):
        p = sim.Population(23, sim.EIF_cond_exp_isfa_ista())
        pv1 = p[1, 5, 6, 8, 11, 12, 15, 16, 19, 20]

        pv2 = pv1[(3, 5, 7)]
        self.assertEqual(pv2.parent, pv1)
        assert_array_almost_equal(pv2.all_cells, p.all_cells[[8, 12, 16]])

    def test__getitem__invalid(self):
        p = sim.Population(23, sim.EIF_cond_exp_isfa_ista())
        pv = p[1, 5, 6, 8, 11, 12, 15, 16, 19, 20]
        self.assertRaises(TypeError, pv.__getitem__, "foo")

    def test__len__(self):
        # len(p) should give the global size (all MPI nodes)
        p = sim.Population(77, sim.IF_cond_exp())
        pv = p[1, 5, 6, 8, 11, 12, 15, 16, 19, 20]
        self.assertEqual(len(pv), pv.size, 10)

    def test_iter(self):
        p = sim.Population(33, sim.IF_cond_exp())
        pv = p[1, 5, 6, 8, 11, 12]
        itr = pv.__iter__()
        assert hasattr(itr, "next")
        self.assertEqual(len(list(itr)), 6)

    def test___add__two(self):
        # adding two population views should give an Assembly
        pv1 = sim.Population(6, sim.IF_cond_exp())[2, 3, 5]
        pv2 = sim.Population(17, sim.IF_cond_exp())[4, 2, 16]
        assembly = pv1 + pv2
        self.assertIsInstance(assembly, sim.Assembly)
        self.assertEqual(assembly.populations, [pv1, pv2])

    def test___add__three(self):
        # adding three population views should give an Assembly
        pv1 = sim.Population(6, sim.IF_cond_exp())[0:3]
        pv2 = sim.Population(17, sim.IF_cond_exp())[1, 5, 14]
        pv3 = sim.Population(9, sim.EIF_cond_exp_isfa_ista())[3:8]
        assembly = pv1 + pv2 + pv3
        self.assertIsInstance(assembly, sim.Assembly)
        self.assertEqual(assembly.populations, [pv1, pv2, pv3])

    def test_nearest(self):
        p = sim.Population(13, sim.IF_cond_exp())
        p.positions = numpy.arange(39).reshape((13, 3)).T
        pv = p[0, 2, 5, 11]
        self.assertEqual(pv.nearest((0.0, 1.0, 2.0)), pv[0])
        self.assertEqual(pv.nearest((3.0, 4.0, 5.0)), pv[0])
        self.assertEqual(pv.nearest((36.0, 37.0, 38.0)), pv[3])
        self.assertEqual(pv.nearest((1.49, 2.49, 3.49)), pv[0])
        self.assertEqual(pv.nearest((1.51, 2.51, 3.51)), pv[0])

    def test_sample(self):
        p = sim.Population(13, sim.IF_cond_exp())
        pv1 = p[0, 3, 7, 10, 12]

        rng = Mock()
        rng.permutation = Mock(return_value=numpy.array([3, 1, 0, 2, 4]))
        pv2 = pv1.sample(3, rng=rng)
        assert_array_equal(pv2.all_cells,
                           p.all_cells[[10, 3, 0]])

    def test_get_multiple_homogeneous_params_with_gather(self):
        p = sim.Population(10, sim.IF_cond_exp, {'tau_m': 12.3, 'cm': 0.2, 'i_offset': 0.})
        pv = p[3:7]
        cm, tau_m = pv.get(('cm', 'tau_m'), gather=True)
        self.assertEqual(cm, 0.987)
        self.assertEqual(tau_m, 12.3)

    def test_get_single_homogeneous_param_with_gather(self):
        p = sim.Population(4, sim.IF_cond_exp, {'tau_m': 12.3, 'cm': 0.2, 'i_offset': 0.})
        pv = p[:]
        cm = pv.get('cm', gather=True)
        self.assertEqual(cm, 0.987)

    def test_get_multiple_inhomogeneous_params_with_gather(self):
        p = sim.Population(4, sim.IF_cond_exp(tau_m=[12.1, 12.2, 12.3, 12.4],
                                              cm=0.2,
                                              i_offset=lambda i: 0. * i))
        pv = p[0, 1, 3]
        cm, tau_m, i_offset = pv.get(('cm', 'tau_m', 'i_offset'), gather=True)
        self.assertIsInstance(cm, float)
        self.assertIsInstance(tau_m, numpy.ndarray)
        assert_array_equal(tau_m, numpy.array([12.1, 12.2, 12.4]))
        self.assertEqual(cm, 0.2)
        assert_array_almost_equal(i_offset, numpy.array([0., 0., 0.]), decimal=12)

    def test_get_sequence_param(self):
        p = sim.Population(3, sim.SpikeSourceArray,
                           {'spike_times': [Sequence([1, 2, 3, 4]),
                                            Sequence([2, 3, 4, 5]),
                                            Sequence([3, 4, 5, 6])]})
        pv = p[1:]
        spike_times = pv.get('spike_times')
        self.assertEqual(spike_times.size, 2)
        assert_array_equal(spike_times[1], Sequence([3, 4, 5, 6]))

    def test_set(self):
        p = sim.Population(4, sim.IF_cond_exp, {'tau_m': 12.3, 'cm': 0.2, 'i_offset': 0.0})
        pv = p[:3]
        rng = MockRNG(start=1.21, delta=0.01, parallel_safe=True)
        pv.set(cm=0.2, tau_m=random.RandomDistribution('uniform', (9.8, 12.3), rng=rng))
        tau_m, cm, i_offset = p.get(('tau_m', 'cm', 'i_offset'), gather=True)
        assert_array_equal(cm, numpy.array([0.2, 0.2, 0.2, 0.2]))
        assert_array_equal(tau_m, numpy.array([9.87, 9.87, 9.87, 12.3]))
        assert_array_equal(i_offset, 0. * numpy.zeros((4,)))

        tau_m, cm, i_offset = pv.get(('tau_m', 'cm', 'i_offset'), gather=True)
        assert_array_equal(cm, numpy.array([0.2, 0.2, 0.2]))
        assert_array_equal(tau_m, numpy.array([9.87, 9.87, 9.87]))
        assert_array_equal(i_offset, 0. * numpy.ones((3,)))

    def test_set_invalid_name(self):
        p = sim.Population(9, sim.EIF_cond_exp_isfa_ista())
        pv = p[3:5]
        self.assertRaises(errors.NonExistentParameterError, pv.set, foo=13.2)

    def test_set_invalid_type(self):
        p = sim.Population(9, sim.IF_cond_exp())
        pv = p[::3]
        self.assertRaises(errors.InvalidParameterValueError, pv.set, tau_m={})
        self.assertRaises(errors.InvalidParameterValueError, pv.set, v_reset='bar')

    def test_set_sequence(self):
        p = sim.Population(5, sim.SpikeSourceArray())
        pv = p[0, 2, 4]
        pv.set(spike_times=[Sequence([1, 2, 3, 4]),
                           Sequence([2, 3, 4, 5]),
                           Sequence([3, 4, 5, 6])])
        spike_times = p.get('spike_times', gather=True)
        self.assertEqual(spike_times.size, 5)
        assert_array_equal(spike_times[1], Sequence([]))
        assert_array_equal(spike_times[2], Sequence([2, 3, 4, 5]))

    def test_set_array(self):
        p = sim.Population(5, sim.IF_cond_exp, {'v_thresh': -54.3})
        pv = p[2:]
        pv.set(v_thresh=-50.0 + numpy.arange(3))
        assert_array_equal(p.get('v_thresh', gather=True),
                           numpy.array([-54.3, -54.3, -50.0, -49.0, -48.0]))

    def test_tset(self):
        p = sim.Population(17, sim.EIF_cond_exp_isfa_ista())
        pv = p[::4]
        pv.set = Mock()
        tau_m = numpy.linspace(10.0, 20.0, num=pv.size)
        pv.tset("tau_m", tau_m)
        pv.set.assert_called_with(tau_m=tau_m)

    def test_rset(self):
        p = sim.Population(17, sim.EIF_cond_exp_isfa_ista())
        pv = p[::4]
        pv.set = Mock()
        v_rest = random.RandomDistribution('uniform', low=-70.0, high=-60.0)
        pv.rset("v_rest", v_rest)
        pv.set.assert_called_with(v_rest=v_rest)

    def test_can_record(self):
        pv = sim.Population(17, sim.EIF_cond_exp_isfa_ista())[::2]
        assert pv.can_record('v')
        assert not pv.can_record('w')
        assert not pv.can_record('gsyn_inh')
        assert pv.can_record('spikes')
        assert not pv.can_record('foo')

    def test_record_with_single_variable(self):
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        pv = p[0, 4, 6, 13]
        pv.record('v')
        sim.run(12.3)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignals), 1)
        n_values = int(round(12.3 / sim.get_time_step())) + 1
        self.assertEqual(data.analogsignals[0].name, 'v')
        self.assertEqual(data.analogsignals[0].shape, (n_values, pv.size))

    def test_record_with_multiple_variables(self):
        p = sim.Population(4, sim.EIF_cond_exp_isfa_ista())
        pv = p[0, 3]
        pv.record(('v', 'spikes'))
        sim.run(10.0)
        data = p.get_data(gather=True).segments[0]
        self.assertEqual(len(data.analogsignals), 1)
        n_values = int(round(10.0 / sim.get_time_step())) + 1
        names = set(arr.name for arr in data.analogsignals)
        self.assertEqual(names, set(('v')))
        for arr in data.analogsignals:
            self.assertEqual(arr.shape, (n_values, pv.size))

    def test_record_v(self):
        pv = sim.Population(2, sim.EIF_cond_exp_isfa_ista())[0:1]
        pv.record = Mock()
        pv.record_v("arg1")
        pv.record.assert_called_with('v', "arg1")

    def test_record_gsyn(self):
        pv = sim.Population(2, sim.EIF_cond_exp_isfa_ista())[1:]
        pv.record = Mock()
        pv.record_gsyn("arg1")
        pv.record.assert_called_with(['gsyn_exc', 'gsyn_inh'], "arg1")

    def test_record_invalid_variable(self):
        pv = sim.Population(14, sim.EIF_cond_exp_isfa_ista())[::3]
        self.assertRaises(errors.RecordingError,
                          pv.record, ('v', 'gsyn_exc'))  # can't record gsyn_exc from this celltype

    def test_get_spike_counts(self):
        p = sim.Population(5, sim.EIF_cond_exp_isfa_ista())
        pv = p[0, 1, 4]
        pv.record('spikes')
        sim.run(100.0)
        self.assertEqual(p.get_spike_counts(),
                         {p.all_cells[0]: 2,
                          p.all_cells[1]: 2,
                          p.all_cells[4]: 2})

    def test_mean_spike_count(self):
        p = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        pv = p[2::3]
        pv.record('spikes')
        sim.run(100.0)
        self.assertEqual(p.mean_spike_count(), 2.0)

    def test_inject(self):
        pv = sim.Population(3, sim.EIF_cond_exp_isfa_ista())[1, 2]
        cs = Mock()
        pv.inject(cs)
        meth, args, kwargs = cs.method_calls[0]
        self.assertEqual(meth, "inject_into")
        self.assertEqual(args, (pv,))

    def test_inject_into_invalid_celltype(self):
        pv = sim.Population(3, sim.SpikeSourceArray())[:2]
        self.assertRaises(TypeError, pv.inject, Mock())

    # test describe method
    def test_describe(self):
        pv = sim.Population(11, sim.IF_cond_exp())[::4]
        self.assertIsInstance(pv.describe(), basestring)
        self.assertIsInstance(pv.describe(template=None), dict)

if __name__ == "__main__":
    unittest.main()
