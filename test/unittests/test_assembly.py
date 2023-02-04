"""
Tests of the common implementation of the Assembly class, using the pyNN.mock
backend.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import unittest
import numpy as np
import sys
import quantities as pq
from numpy.testing import assert_array_equal, assert_array_almost_equal
try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch
from .mocks import MockRNG
import pyNN.mock as sim
from pyNN.parameters import Sequence


def setUp():
    pass


def tearDown():
    pass


class AssemblyTest(unittest.TestCase):

    def setUp(self, sim=sim, **extra):
        sim.setup(**extra)

    def tearDown(self, sim=sim):
        sim.end()

    def test_create_with_zero_populations(self, sim=sim):
        a = sim.Assembly()
        self.assertEqual(a.populations, [])
        self.assertIsInstance(a.label, str)

    def test_create_with_one_population(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p)
        self.assertEqual(a.populations, [p])
        self.assertIsInstance(a.label, str)

    def test_create_with_two_populations(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        self.assertEqual(a.populations, [p1, p2])
        self.assertEqual(a.label, "test")

    def test_create_with_non_population_should_raise_Exception(self, sim=sim):
        self.assertRaises(TypeError, sim.Assembly, [1, 2, 3])

    def test_size_property(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        self.assertEqual(a.size, p1.size + p2.size)

    def test_positions_property(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assert_array_equal(a.positions, np.concatenate((p1.positions, p2.positions), axis=1))

    def test__len__(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        self.assertEqual(len(a), len(p1) + len(p2))

    def test_local_cells(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assert_array_equal(a.local_cells, np.append(p1.local_cells, p2.local_cells))

    def test_all_cells(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assert_array_equal(a.all_cells, np.append(p1.all_cells, p2.all_cells))

    def test_iter(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assembly_ids = [id for id in a]

    def test__add__population(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a1 = sim.Assembly(p1)
        self.assertEqual(a1.populations, [p1])
        a2 = a1 + p2
        self.assertEqual(a1.populations, [p1])
        self.assertEqual(a2.populations, [p1, p2])

    def test__add__assembly(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a1 = sim.Assembly(p1, p2)
        a2 = sim.Assembly(p2, p3)
        a3 = a1 + a2
        self.assertEqual(a3.populations, [p1, p2, p3])  # or do we want [p1, p2, p3]?

    def test_add_inplace_population(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1)
        a += p2
        self.assertEqual(a.populations, [p1, p2])

    def test_add_inplace_assembly(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a1 = sim.Assembly(p1, p2)
        a2 = sim.Assembly(p2, p3)
        a1 += a2
        self.assertEqual(a1.populations, [p1, p2, p3])

    def test_add_invalid_object(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2)
        self.assertRaises(TypeError, a.__add__, 42)
        self.assertRaises(TypeError, a.__iadd__, 42)

    def test_initialize(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2)
        v_init = -54.3
        a.initialize(v=v_init)
        assert_array_equal(p2.initial_values['v'].evaluate(simplify=True), v_init)

    def test_describe(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2)
        self.assertIsInstance(a.describe(), str)
        self.assertIsInstance(a.describe(template=None), dict)

    def test_get_population(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p1.label = "pop1"
        p2 = sim.Population(11, sim.IF_cond_exp())
        p2.label = "pop2"
        a = sim.Assembly(p1, p2)
        self.assertEqual(a.get_population("pop1"), p1)
        self.assertEqual(a.get_population("pop2"), p2)
        self.assertRaises(KeyError, a.get_population, "foo")

    def test_all_cells(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertEqual(a.all_cells.size,
                         p1.all_cells.size + p2.all_cells.size + p3.all_cells.size)
        self.assertEqual(a.all_cells[0], p1.all_cells[0])
        self.assertEqual(a.all_cells[-1], p3.all_cells[-1])
        assert_array_equal(a.all_cells, np.append(p1.all_cells, (p2.all_cells, p3.all_cells)))

    def test_local_cells(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertEqual(a.local_cells.size,
                         p1.local_cells.size + p2.local_cells.size + p3.local_cells.size)
        self.assertEqual(a.local_cells[0], p1.local_cells[0])
        self.assertEqual(a.local_cells[-1], p3.local_cells[-1])
        assert_array_equal(a.local_cells, np.append(
            p1.local_cells, (p2.local_cells, p3.local_cells)))

    def test_mask_local(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertEqual(a._mask_local.size,
                         p1._mask_local.size + p2._mask_local.size + p3._mask_local.size)
        self.assertEqual(a._mask_local[0], p1._mask_local[0])
        self.assertEqual(a._mask_local[-1], p3._mask_local[-1])
        assert_array_equal(a._mask_local, np.append(
            p1._mask_local, (p2._mask_local, p3._mask_local)))
        assert_array_equal(a.local_cells, a.all_cells[a._mask_local])

    def test_save_positions(self, sim=sim):
        import os
        p1 = sim.Population(2, sim.IF_cond_exp())
        p2 = sim.Population(2, sim.IF_cond_exp())
        p1.positions = np.arange(0, 6).reshape((2, 3)).T
        p2.positions = np.arange(6, 12).reshape((2, 3)).T
        a = sim.Assembly(p1, p2, label="test")
        output_file = Mock()
        a.save_positions(output_file)
        assert_array_equal(output_file.write.call_args[0][0],
                           np.array([[0, 0, 1, 2],
                                        [1, 3, 4, 5],
                                        [2, 6, 7, 8],
                                        [3, 9, 10, 11]]))
        self.assertEqual(output_file.write.call_args[0][1], {'assembly': a.label})

    def test_repr(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertIsInstance(repr(a), str)

    def test_ids_should_not_be_counted_twice(self, sim=sim):
        p = sim.Population(11, sim.IF_cond_exp())
        pv1 = p[0:5]
        a1 = sim.Assembly(p, pv1)
        self.assertEqual(a1.size, p.size)
        #a2 = sim.Assembly(pv1, p)
        #self.assertEqual(a2.size, p.size)
        #pv2 = p[3:7]
        #a3 = sim.Assembly(pv1, pv2)
        #self.assertEqual(a3.size, 7)

    def test_all_iterator(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p1, p2, p3)
        assert hasattr(a.all(), "next") or hasattr(a.all(), "__next__")  # 2nd form is for Py3
        ids = list(a.all())
        self.assertEqual(ids, p1.all_cells.tolist() +
                         p2.all_cells.tolist() + p3.all_cells.tolist())

    def test__homogeneous_synapses(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        a1 = sim.Assembly(p1, p2)
        self.assertTrue(a1._homogeneous_synapses)

    def test__non_homogeneous_synapses(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a2 = sim.Assembly(p1, p3)
        self.assertFalse(a2._homogeneous_synapses)

    def test_conductance_based(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        a1 = sim.Assembly(p1, p2)
        self.assertTrue(a1.conductance_based)

    def test_not_conductance_based(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a2 = sim.Assembly(p1, p3)
        self.assertFalse(a2.conductance_based)

    def test_first_and_last_id(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        self.assertEqual(a.first_id, p1[0])
        self.assertEqual(a.last_id, p3[-1])

    def test_id_to_index(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        self.assertEqual(a.id_to_index(p3[0]), 0)
        self.assertEqual(a.id_to_index(p1[0]), 3)
        self.assertEqual(a.id_to_index(p2[0]), 14)
        assert_array_equal(a.id_to_index([p1[0], p2[0], p3[0]]), [3, 14, 0])

    def test_id_to_index_with_nonexistent_id(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        self.assertRaises(IndexError, a.id_to_index, p3.last_id + 1)

    def test_getitem_int(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        self.assertEqual(a[0], p3[0])
        self.assertEqual(a[3], p1[0])
        self.assertEqual(a[14], p2[0])

    def test_getitem_slice(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        a1 = a[0:3]
        self.assertIsInstance(a1, sim.Assembly)
        self.assertEqual(len(a1.populations), 1)
        assert_array_equal(a1.populations[0].all_cells, p3[:].all_cells)
        a2 = a[2:8]
        self.assertEqual(len(a2.populations), 2)
        assert_array_equal(a2.populations[0].all_cells, p3[2:].all_cells)
        assert_array_equal(a2.populations[1].all_cells, p1[:5].all_cells)

    def test_getitem_array(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        a1 = a[3, 5, 6, 10]
        self.assertIsInstance(a1, sim.Assembly)
        self.assertEqual(len(a1.populations), 1)
        assert_array_equal(a1.populations[0].all_cells, [p1[0], p1[2], p1[3], p1[7]])
        a2 = a[17, 2, 10, 11, 19]
        self.assertEqual(len(a2.populations), 3)
        assert_array_equal(a2.populations[0].all_cells, p3[2:].all_cells)
        assert_array_equal(a2.populations[1].all_cells, p1[7:9].all_cells)
        assert_array_equal(a2.populations[2].all_cells, p2[3, 5].all_cells)

    def test_sample(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.IF_curr_exp())
        a = sim.Assembly(p3, p1, p2)
        a1 = a.sample(10, rng=MockRNG())
        # MockRNG.permutation reverses the order
        self.assertEqual(len(a1.populations), 2)
        assert_array_equal(a1.populations[0].all_cells, sorted(p1[11:6:-1]))
        assert_array_equal(a1.populations[1].all_cells, sorted(p2[6::-1]))

    def test_get_data_with_gather(self, sim=sim):
        t1 = 12.3
        t2 = 13.4
        t3 = 14.5
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_alpha())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        a.record('v')
        sim.run(t1)
        # what if we call p.record between two run statements?
        # would be nice to get an AnalogSignal with a non-zero t_start
        # but then need to make sure we get the right initial value
        sim.run(t2)
        sim.reset()
        a.record('spikes')
        p3.record('w')
        sim.run(t3)
        data = a.get_data(gather=True)
        self.assertEqual(len(data.segments), 2)
        seg0 = data.segments[0]
        self.assertEqual(len(seg0.analogsignals), 1)
        v = seg0.filter(name='v')[0]
        self.assertEqual(v.name, 'v')
        num_points = int(round((t1 + t2) / sim.get_time_step())) + 1
        self.assertEqual(v.shape, (num_points, a.size))
        self.assertEqual(v.t_start, 0.0 * pq.ms)
        self.assertEqual(v.units, pq.mV)
        self.assertEqual(v.sampling_period, 0.1 * pq.ms)
        self.assertEqual(len(seg0.spiketrains), 0)

        seg1 = data.segments[1]
        self.assertEqual(len(seg1.analogsignals), 2)
        w = seg1.filter(name='w')[0]
        self.assertEqual(w.name, 'w')
        num_points = int(round(t3 / sim.get_time_step())) + 1
        self.assertEqual(w.shape, (num_points, p3.size))
        self.assertEqual(v.t_start, 0.0)
        self.assertEqual(len(seg1.spiketrains), a.size)
        # assert_array_equal(seg1.spiketrains[7],
        #                   np.array([a.first_id+7, a.first_id+7+5]) % t3)

    def test_printSpikes(self, sim=sim):
        # TODO: implement assert_deprecated
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record('spikes')
        sim.run(10.0)
        a.write_data = Mock()
        a.printSpikes("foo.txt")
        a.write_data.assert_called_with('foo.txt', 'spikes', True)

    def test_getSpikes(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record('spikes')
        sim.run(10.0)
        a.get_data = Mock()
        a.getSpikes()
        a.get_data.assert_called_with('spikes', True)

    def test_print_v(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_v()
        sim.run(10.0)
        a.write_data = Mock()
        a.print_v("foo.txt")
        a.write_data.assert_called_with('foo.txt', 'v', True)

    def test_get_v(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_v()
        sim.run(10.0)
        a.get_data = Mock()
        a.get_v()
        a.get_data.assert_called_with('v', True)

    def test_print_gsyn(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_gsyn()
        sim.run(10.0)
        a.write_data = Mock()
        a.print_gsyn("foo.txt")
        a.write_data.assert_called_with('foo.txt', ['gsyn_exc', 'gsyn_inh'], True)

    def test_get_gsyn(self, sim=sim):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_gsyn()
        sim.run(10.0)
        a.get_data = Mock()
        a.get_gsyn()
        a.get_data.assert_called_with(['gsyn_exc', 'gsyn_inh'], True)

    def test_get_multiple_homogeneous_params_with_gather(self, sim=sim):
        p1 = sim.Population(4, sim.IF_cond_exp(
            **{'tau_m': 12.3, 'tau_syn_E': 0.987, 'tau_syn_I': 0.7}))
        p2 = sim.Population(4, sim.IF_curr_exp(
            **{'tau_m': 12.3, 'tau_syn_E': 0.987, 'tau_syn_I': 0.7}))
        a = p1 + p2
        tau_syn_E, tau_m = a.get(('tau_syn_E', 'tau_m'), gather=True)
        self.assertIsInstance(tau_syn_E, float)
        self.assertEqual(tau_syn_E, 0.987)
        self.assertAlmostEqual(tau_m, 12.3)

    def test_get_single_param_with_gather(self, sim=sim):
        p1 = sim.Population(4, sim.IF_cond_exp(tau_m=12.3, tau_syn_E=0.987, tau_syn_I=0.7))
        p2 = sim.Population(3, sim.IF_cond_exp(tau_m=23.4, tau_syn_E=0.987, tau_syn_I=0.7))
        a = p1 + p2
        tau_syn_E = a.get('tau_syn_E', gather=True)
        self.assertAlmostEqual(tau_syn_E, 0.987, places=6)
        tau_m = a.get('tau_m', gather=True)
        assert_array_equal(tau_m, np.array([12.3, 12.3, 12.3, 12.3, 23.4, 23.4, 23.4]))

    def test_get_multiple_inhomogeneous_params_with_gather(self, sim=sim):
        p1 = sim.Population(4, sim.IF_cond_exp(tau_m=12.3,
                                               tau_syn_E=[0.987, 0.988, 0.989, 0.990],
                                               tau_syn_I=lambda i: 0.5 + 0.1 * i))
        p2 = sim.Population(3, sim.EIF_cond_exp_isfa_ista(tau_m=12.3,
                                                          tau_syn_E=[0.991, 0.992, 0.993],
                                                          tau_syn_I=lambda i: 0.5 + 0.1 * (i + 4)))
        a = p1 + p2
        tau_syn_E, tau_m, tau_syn_I = a.get(('tau_syn_E', 'tau_m', 'tau_syn_I'), gather=True)
        self.assertIsInstance(tau_m, float)
        self.assertIsInstance(tau_syn_E, np.ndarray)
        assert_array_equal(tau_syn_E, np.array(
            [0.987, 0.988, 0.989, 0.990, 0.991, 0.992, 0.993]))
        self.assertAlmostEqual(tau_m, 12.3)
        assert_array_almost_equal(tau_syn_I, np.array(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]), decimal=12)

    def test_get_multiple_params_no_gather(self, sim=sim):
        sim.simulator.state.num_processes = 2
        sim.simulator.state.mpi_rank = 1
        p1 = sim.Population(4, sim.IF_cond_exp(tau_m=12.3,
                                               tau_syn_E=[0.987, 0.988, 0.989, 0.990],
                                               i_offset=lambda i: -0.2 * i))
        p2 = sim.Population(3, sim.IF_curr_exp(tau_m=12.3,
                                               tau_syn_E=[0.991, 0.992, 0.993],
                                               i_offset=lambda i: -0.2 * (i + 4)))
        a = p1 + p2
        tau_syn_E, tau_m, i_offset = a.get(('tau_syn_E', 'tau_m', 'i_offset'), gather=False)
        self.assertIsInstance(tau_m, float)
        self.assertIsInstance(tau_syn_E, np.ndarray)
        assert_array_equal(tau_syn_E, np.array([0.988, 0.990, 0.992]))
        self.assertEqual(tau_m, 12.3)
        assert_array_almost_equal(i_offset, np.array([-0.2, -0.6, -1.0, ]), decimal=12)
        sim.simulator.state.num_processes = 1
        sim.simulator.state.mpi_rank = 0

    def test_get_sequence_param(self, sim=sim):
        p1 = sim.Population(3, sim.SpikeSourceArray(spike_times=[Sequence([1, 2, 3, 4]),
                                                                 Sequence([2, 3, 4, 5]),
                                                                 Sequence([3, 4, 5, 6])]))
        p2 = sim.Population(2, sim.SpikeSourceArray(spike_times=[Sequence([4, 5, 6, 7]),
                                                                 Sequence([5, 6, 7, 8])]))
        a = p1 + p2
        spike_times = a.get('spike_times')
        self.assertEqual(spike_times.size, 5)
        assert_array_equal(spike_times[3], Sequence([4, 5, 6, 7]))

    def test_inject(self, sim=sim):
        p1 = sim.Population(3, sim.IF_curr_alpha())
        p2 = sim.Population(5, sim.IF_cond_exp())
        a = p1 + p2
        cs = Mock()
        a.inject(cs)
        meth, args, kwargs = cs.method_calls[0]
        self.assertEqual(meth, "inject_into")
        self.assertEqual(args, (p1,))
        meth, args, kwargs = cs.method_calls[1]
        self.assertEqual(meth, "inject_into")
        self.assertEqual(args, (p2,))

    def test_mean_spike_count(self, sim=sim):
        p1 = sim.Population(14, sim.EIF_cond_exp_isfa_ista())
        p2 = sim.Population(37, sim.IF_cond_alpha())
        a = p1 + p2
        p1.record('spikes')
        p2.record('spikes')
        # a.record('spikes')
        sim.run(100.0)
        # mock backend always produces two spikes per population
        self.assertEqual(a.mean_spike_count(), 2.0)


if __name__ == '__main__':
    unittest.main()
