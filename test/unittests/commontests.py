"""
Unit tests for the common module
$Id:$
"""

import sys
import unittest
import numpy
import os
from pyNN import common

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) < threshold).all()

# ==============================================================================
class ExceptionsTest(unittest.TestCase):
    
    def test_NonExistentParameterError_withStandardModel(self):
        try:
            raise common.NonExistentParameterError("foo", common.IF_cond_alpha)
        except common.NonExistentParameterError, err:
            self.assertEqual(err.model_name, 'IF_cond_alpha')
            self.assertEqual(err.parameter_name, 'foo')
            self.assertEqual(err.valid_parameter_names,
                             ['cm', 'e_rev_E', 'e_rev_I', 'i_offset', 'tau_m',
                              'tau_refrac', 'tau_syn_E', 'tau_syn_I', 'v_init',
                              'v_reset', 'v_rest', 'v_thresh'])
            assert len(str(err)) > 0
    
    def test_NonExistentParameterError_withStringModel(self):
        try:
            raise common.NonExistentParameterError("foo", 'iaf_neuron')
        except common.NonExistentParameterError, err:
            self.assertEqual(err.model_name, 'iaf_neuron')
            self.assertEqual(err.parameter_name, 'foo')
            self.assertEqual(err.valid_parameter_names, ['unknown'])
            assert len(str(err)) > 0
    
    def test_NonExistentParameterError_withInvalidModel(self):
        # model must be a string or a class
        self.assertRaises(Exception, common.NonExistentParameterError, "foo", [])
    
    
class NotImplementedTest(unittest.TestCase):
    
    def testPopulationStubs(self):
        p = common.Population(10, common.IF_cond_alpha)
        for method_name in ('__iter__', 'addresses', 'ids'):
            self.assertRaises(NotImplementedError, getattr(p, method_name))
        for method_name in ('locate', 'index'):
            self.assertRaises(NotImplementedError, getattr(p, method_name), 0)
        for method_name in ('set', 'tset', 'rset'):
            self.assertRaises(NotImplementedError, getattr(p, method_name), 'tau_m', 'dummy_value')
        for method_name in ('_call', '_tcall'):
            self.assertRaises(NotImplementedError, getattr(p, method_name), 'do_foo', 'dummy_value')
        for method_name in ('record', 'record_v', 'getSpikes', 'meanSpikeCount'):
            self.assertRaises(NotImplementedError, getattr(p, method_name))
        for method_name in ('printSpikes', 'print_v'):
            self.assertRaises(NotImplementedError, getattr(p, method_name), 'filename')
        for method_name in ('getSubPopulation',):
            self.assertRaises(NotImplementedError, getattr(p, method_name), [])
        
    def testProjectionStubs(self):
        p = common.Population(10, common.IF_cond_alpha)
        prj = common.Projection(p, p, common.AllToAllConnector)
        for method_name in ('setWeights', 'randomizeWeights', 'setDelays', 'randomizeDelays'):
            self.assertRaises(NotImplementedError, getattr(prj, method_name), 'dummy_value')
        for method_name in ('setSynapseDynamics', 'randomizeSynapseDynamics'):
            self.assertRaises(NotImplementedError, getattr(prj, method_name), 'foo', 'dummy_value')
        for method_name in ('getWeights', 'getDelays'):
            self.assertRaises(NotImplementedError, getattr(prj, method_name))                      
        for method_name in ('saveConnections',):
            self.assertRaises(NotImplementedError, getattr(prj, method_name), 'filename')
        
        
class DistanceTest(unittest.TestCase):
    
    def testDistance(self):
        cell1 = common.IDMixin()
        cell2 = common.IDMixin()
        cell1.position = (2.3, 4.5, 6.7)
        cell2.position = (2.3, 4.5, 6.7)
        self.assertEqual(common.distance(cell1, cell2), 0.0)
        cell2.position = (5.3, 4.5, 6.7)
        self.assertEqual(common.distance(cell1, cell2), 3.0)
        cell2.position = (5.3, 8.5, 6.7)
        self.assertEqual(common.distance(cell1, cell2), 5.0)
        cell2.position = (5.3, 8.5, -5.3)
        self.assertEqual(common.distance(cell1, cell2), 13.0)
        self.assertEqual(common.distance(cell1, cell2, mask=numpy.array([0,1])), 5.0)
        self.assertEqual(common.distance(cell1, cell2, mask=numpy.array([2])), 12.0)
        self.assertEqual(common.distance(cell1, cell2, offset=numpy.array([-3.0, -4.0, 12.0])), 0.0)
        cell2.position = (10.6, 17.0, -10.6)
        self.assertEqual(common.distance(cell1, cell2, scale_factor=0.5), 13.0)
        cell2.position = (-1.7, 8.5, -5.3)
        self.assertEqual(common.distance(cell1, cell2, periodic_boundaries=numpy.array([7.0, 1e12, 1e12])), 13.0)
        
class StandardModelTest(unittest.TestCase):
    
    def testCheckParameters(self):
        self.assertRaises(common.InvalidParameterValueError, common.SpikeSourceArray, {'spike_times': 0.0})
        self.assertRaises(common.InvalidParameterValueError, common.SpikeSourceInhGamma, {'a': 'foo'})
        self.assertRaises(ValueError, common.SpikeSourceArray, {'spike_times': 'foo'})
        self.assertRaises(common.NonExistentParameterError, common.IF_cond_exp, {'foo': 'bar'})
        
    def testTranslate(self):
        class FakeCellType(common.StandardCellType):
            default_parameters = {'foo': 3, 'other_parameter': 5}
        class SimFakeCellType1(FakeCellType):
            translations = common.build_translations(('foo', 'bar', 'foo*non_existent_parameter', 'bar/non_existent_parameter'))
        class SimFakeCellType2(FakeCellType):
            translations = common.build_translations(
                ('foo', 'bar', 'foo*other_parameter', 'bar/non_existent_parameter'),
                ('other_parameter', 'translated_other_parameter'),    
            )
        self.assertRaises(NameError, SimFakeCellType1, {'foo': 4})
        cell_type = SimFakeCellType2({'foo': 4, 'other_parameter': 5})
        self.assertRaises(NameError, cell_type.reverse_translate, {'bar': 20})
        assert isinstance(cell_type.describe(), basestring) # this belongs in a separate test
        
    def testCreatingNonAvailableModel(self):
        self.assertRaises(NotImplementedError, common.ModelNotAvailable)
                              
# ==============================================================================
if __name__ == "__main__":
    unittest.main()