"""
Tests of the common implementation of the Assembly class, using the pyNN.mock
backend.

:copyright: Copyright 2006-2019 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy
import quantities as pq
from numpy.testing import assert_array_equal, assert_array_almost_equal
from mock import Mock, patch
from .mocks import MockRNG
import pyNN.hardware.brainscales as sim


class AssemblyTest(unittest.TestCase):

    def setUp(self):
        extra = {'loglevel': 0, 'useSystemSim': True, 'hardware': sim.hardwareSetup['one-hicann']}
        sim.setup(**extra)
        
    def tearDown(self):
        sim.end()

    def test_create_with_zero_populations(self):
        a = sim.Assembly()
        self.assertEqual(a.populations, [])
        self.assertIsInstance(a.label, basestring)

    def test_create_with_one_population(self):
        p = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p)
        self.assertEqual(a.populations, [p])
        self.assertIsInstance(a.label, basestring)

    def test_create_with_two_populations(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        self.assertEqual(a.populations, [p1, p2])
        self.assertEqual(a.label, "test")
    
    def test_create_with_non_population_should_raise_Exception(self):
        self.assertRaises(TypeError, sim.Assembly, [1, 2, 3])
    
    def test_size_property(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        self.assertEqual(a.size, p1.size + p2.size)
    
    def test_positions_property(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assert_array_equal(a.positions, numpy.concatenate((p1.positions, p2.positions), axis=1))
    
    def test__len__(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        self.assertEqual(len(a), len(p1) + len(p2))
    
    def test_local_cells(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assert_array_equal(a.local_cells, numpy.append(p1.local_cells, p2.local_cells))
    
    def test_all_cells(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assert_array_equal(a.all_cells, numpy.append(p1.all_cells, p2.all_cells))
    
    def test_iter(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        assembly_ids = [id for id in a]
    
    def test__add__population(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a1 = sim.Assembly(p1)
        self.assertEqual(a1.populations, [p1])
        a2 = a1 + p2
        self.assertEqual(a1.populations, [p1])
        self.assertEqual(a2.populations, [p1, p2])
    
    def test__add__assembly(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a1 = sim.Assembly(p1, p2)
        a2 = sim.Assembly(p2, p3)
        a3 = a1 + a2
        self.assertEqual(a3.populations, [p1, p2, p3])  # or do we want [p1, p2, p3]?
    
    def test_add_inplace_population(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1)
        a += p2
        self.assertEqual(a.populations, [p1, p2])
    
    def test_add_inplace_assembly(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a1 = sim.Assembly(p1, p2)
        a2 = sim.Assembly(p2, p3)
        a1 += a2
        self.assertEqual(a1.populations, [p1, p2, p3])
    
    def test_add_invalid_object(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2)
        self.assertRaises(TypeError, a.__add__, 42)
        self.assertRaises(TypeError, a.__iadd__, 42)
    
    def test_initialize(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2)
        v_init = -54.3
        a.initialize(v=v_init)
        assert_array_equal(p2.initial_values['v'].evaluate(simplify=True), v_init)
    
    def test_describe(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2)
        self.assertIsInstance(a.describe(), basestring)
        self.assertIsInstance(a.describe(template=None), dict)
    
    def test_get_population(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p1.label = "pop1"
        p2 = sim.Population(11, sim.IF_cond_exp())
        p2.label = "pop2"
        a = sim.Assembly(p1, p2)
        self.assertEqual(a.get_population("pop1"), p1)
        self.assertEqual(a.get_population("pop2"), p2)
        self.assertRaises(KeyError, a.get_population, "foo")
    
    def test_all_cells(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertEqual(a.all_cells.size,
                     p1.all_cells.size + p2.all_cells.size + p3.all_cells.size)
        self.assertEqual(a.all_cells[0], p1.all_cells[0])
        self.assertEqual(a.all_cells[-1], p3.all_cells[-1])
        assert_array_equal(a.all_cells, numpy.append(p1.all_cells, (p2.all_cells, p3.all_cells)))
    
    def test_local_cells(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertEqual(a.local_cells.size,
                     p1.local_cells.size + p2.local_cells.size + p3.local_cells.size)
        self.assertEqual(a.local_cells[0], p1.local_cells[0])
        self.assertEqual(a.local_cells[-1], p3.local_cells[-1])
        assert_array_equal(a.local_cells, numpy.append(p1.local_cells, (p2.local_cells, p3.local_cells)))
    
    def test_mask_local(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertEqual(a._mask_local.size,
                     p1._mask_local.size + p2._mask_local.size + p3._mask_local.size)
        self.assertEqual(a._mask_local[0], p1._mask_local[0])
        self.assertEqual(a._mask_local[-1], p3._mask_local[-1])
        assert_array_equal(a._mask_local, numpy.append(p1._mask_local, (p2._mask_local, p3._mask_local)))
        assert_array_equal(a.local_cells, a.all_cells[a._mask_local])
    
    def test_save_positions(self):
        import os
        p1 = sim.Population(2, sim.IF_cond_exp())
        p2 = sim.Population(2, sim.IF_cond_exp())
        p1.positions = numpy.arange(0, 6).reshape((2, 3)).T
        p2.positions = numpy.arange(6, 12).reshape((2, 3)).T
        a = sim.Assembly(p1, p2, label="test")
        output_file = Mock()
        a.save_positions(output_file)
        assert_array_equal(output_file.write.call_args[0][0],
                            numpy.array([[int(p1[0]), 0, 1, 2],
                                         [int(p1[1]), 3, 4, 5],
                                         [int(p2[0]), 6, 7, 8],
                                         [int(p2[1]), 9, 10, 11]]))
        self.assertEqual(output_file.write.call_args[0][1], {'assembly': a.label})
        # arguably, the first column should contain indices, not ids.

    def test_repr(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        p3 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, p3)
        self.assertIsInstance(repr(a), str)

    def test_ids_should_not_be_counted_twice(self):
        p = sim.Population(11, sim.IF_cond_exp())
        pv1 = p[0:5]
        a1 = sim.Assembly(p, pv1)
        self.assertEqual(a1.size, p.size)
        #a2 = sim.Assembly(pv1, p)
        #self.assertEqual(a2.size, p.size)
        #pv2 = p[3:7]
        #a3 = sim.Assembly(pv1, pv2)
        #self.assertEqual(a3.size, 7)

    def test_all_iterator(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p1, p2, p3)
        assert hasattr(a.all(), "next")
        ids = list(a.all())
        self.assertEqual(ids, p1.all_cells.tolist() + p2.all_cells.tolist() + p3.all_cells.tolist())

    def test__homogeneous_synapses(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a1 = sim.Assembly(p1, p2)
        a2 = sim.Assembly(p1, p3)
        self.assertTrue(a1._homogeneous_synapses)
        self.assertFalse(a2._homogeneous_synapses)

    def test_conductance_based(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a1 = sim.Assembly(p1, p2)
        a2 = sim.Assembly(p1, p3)
        self.assertTrue(a1.conductance_based)
        self.assertFalse(a2.conductance_based)

    def test_first_and_last_id(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        self.assertEqual(a.first_id, p1[0])
        self.assertEqual(a.last_id, p3[-1])

    def test_id_to_index(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        self.assertEqual(a.id_to_index(p3[0]), 0)
        self.assertEqual(a.id_to_index(p1[0]), 3)
        self.assertEqual(a.id_to_index(p2[0]), 14)
        assert_array_equal(a.id_to_index([p1[0], p2[0], p3[0]]), [3, 14, 0])
    
    def test_id_to_index_with_nonexistent_id(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        self.assertRaises(IndexError, a.id_to_index, p3.last_id + 1)

    def test_getitem_int(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        self.assertEqual(a[0], p3[0])
        self.assertEqual(a[3], p1[0])
        self.assertEqual(a[14], p2[0])

    def test_getitem_slice(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        a1 = a[0:3]
        self.assertIsInstance(a1, sim.Assembly)
        self.assertEqual(len(a1.populations), 1)
        assert_array_equal(a1.populations[0].all_cells, p3[:].all_cells)
        a2 = a[2:8]
        self.assertEqual(len(a2.populations), 2)
        assert_array_equal(a2.populations[0].all_cells, p3[2:].all_cells)
        assert_array_equal(a2.populations[1].all_cells, p1[:5].all_cells)

    def test_getitem_array(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
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

    def test_sample(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(6, sim.IF_cond_exp())
        p3 = sim.Population(3, sim.EIF_cond_exp_isfa_ista())
        a = sim.Assembly(p3, p1, p2)
        a1 = a.sample(10, rng=MockRNG())
        # MockRNG.permutation reverses the order
        self.assertEqual(len(a1.populations), 2)
        assert_array_equal(a1.populations[0].all_cells, p1[11:6:-1])
        assert_array_equal(a1.populations[1].all_cells, p2[6::-1])

    def test_printSpikes(self):
        # TODO: implement assert_deprecated
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record('spikes')
        sim.run(10.0)
        a.write_data = Mock()
        a.printSpikes("foo.txt")
        a.write_data.assert_called_with('foo.txt', 'spikes', True)

    def test_getSpikes(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record('spikes')
        sim.run(10.0)
        a.get_data = Mock()
        a.getSpikes()
        a.get_data.assert_called_with('spikes', True)

    def test_print_v(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_v()
        sim.run(10.0)
        a.write_data = Mock()
        a.print_v("foo.txt")
        a.write_data.assert_called_with('foo.txt', 'v', True)

    def test_get_v(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_v()
        sim.run(10.0)
        a.get_data = Mock()
        a.get_v()
        a.get_data.assert_called_with('v', True)

    def test_print_gsyn(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_gsyn()
        sim.run(10.0)
        a.write_data = Mock()
        a.print_gsyn("foo.txt")
        a.write_data.assert_called_with('foo.txt', ['gsyn_exc', 'gsyn_inh'], True)

    def test_get_gsyn(self):
        p1 = sim.Population(11, sim.IF_cond_exp())
        p2 = sim.Population(11, sim.IF_cond_exp())
        a = sim.Assembly(p1, p2, label="test")
        a.record_gsyn()
        sim.run(10.0)
        a.get_data = Mock()
        a.get_gsyn()
        a.get_data.assert_called_with(['gsyn_exc', 'gsyn_inh'], True)


if __name__ == '__main__':
    unittest.main()
