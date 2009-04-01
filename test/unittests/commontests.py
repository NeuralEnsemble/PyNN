"""
Unit tests for the common module
$Id:$
"""

import sys
import unittest
import numpy
import os
from pyNN import common, random

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) <= threshold).all()

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
        for method_name in ('rset',):
            self.assertRaises(NotImplementedError, getattr(p, method_name), 'tau_m', 'dummy_value')
        for method_name in ('_call', '_tcall'):
            self.assertRaises(NotImplementedError, getattr(p, method_name), 'do_foo', 'dummy_value')
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
                              

class LowLevelAPITest(unittest.TestCase):
    
    def test_setup(self):
        self.assertRaises(Exception, common.setup, min_delay=1.0, max_delay=0.9)
        self.assertRaises(Exception, common.setup, mindelay=1.0)
        self.assertRaises(Exception, common.setup, maxdelay=10.0)
        self.assertRaises(Exception, common.setup, dt=0.1)
        
    def test_end(self):
        self.assertRaises(NotImplementedError, common.end)
        
    def test_run(self):
        self.assertRaises(NotImplementedError, common.run, 10.0)
               
    def test_get_simulator_state(self):
        self.assertRaises(NotImplementedError, common.get_current_time)
        self.assertRaises(NotImplementedError, common.get_time_step)
        self.assertRaises(NotImplementedError, common.get_min_delay)
        self.assertRaises(NotImplementedError, common.get_max_delay)
        self.assertRaises(NotImplementedError, common.num_processes)
        self.assertRaises(NotImplementedError, common.rank)
        
class PopulationTest(unittest.TestCase):
    
    def test_positions(self):
        p = common.Population((4,5,6), common.IF_cond_exp)
        self.assertEqual(p.positions.shape, (3,120))
        self.assertEqual(tuple(p.positions[:,0]), (0.0,0.0,0.0))
        self.assertEqual(tuple(p.positions[:,6]), (0.0,1.0,0.0))
        self.assertEqual(tuple(p.positions[:,30]), (1.0,0.0,0.0))
        pos = numpy.random.random((3,120))
        p.positions = pos
        assert arrays_almost_equal(pos, p.positions, 0.0)
        pos[0,0] = 99.9
        assert not arrays_almost_equal(pos, p.positions, 0.0)
        
    def test_canrecord(self):
        p1 = common.Population((4,5,6), common.IF_cond_exp)
        assert p1.can_record('spikes')
        assert p1.can_record('v')
        assert not p1.can_record('foo')
        p2 = common.Population((4,5,6), common.SpikeSourceArray)
        assert p2.can_record('spikes')
        assert not p2.can_record('v')
        
class ConnectorTest(unittest.TestCase):

    def test_getWeights(self):
        c1 = common.Connector(delays=0.5, weights=0.5)
        self.assertEqual(c1.getWeights(3).tolist(), [0.5,0.5,0.5])
        c2 = common.Connector(delays=0.5, weights="foo")
        self.assertRaises(ValueError, c2.getWeights, 3)
        class A(object): pass
        c3 = common.Connector(delays=0.5, weights=A())
        self.assertRaises(Exception,c3.getWeights, 3)
        rd = random.RandomDistribution('gamma', [0.5,0.5])
        c4 = common.Connector(delays=0.5, weights=rd)
        w = c4.getWeights(3)
        self.assertEqual(len(w), 3)
        self.assertNotEqual(w[0], w[1])
        
    def test_getDelays(self):
        c1 = common.Connector(delays=0.5, weights=0.5)
        self.assertEqual(c1.getDelays(3).tolist(), [0.5,0.5,0.5])
        c2 = common.Connector(weights=0.5, delays="foo")
        self.assertRaises(ValueError, c2.getDelays, 3)
        class A(object): pass
        c3 = common.Connector(weights=0.5, delays=A())
        self.assertRaises(Exception,c3.getDelays, 3)
        rd = random.RandomDistribution('gamma', [0.5,0.5])
        c4 = common.Connector(weights=0.5, delays=rd)
        d = c4.getDelays(3)
        self.assertEqual(len(d), 3)
        self.assertNotEqual(d[0], d[1])
        c5 = common.Connector(weights=0.5, delays=[1.0, 2.0, 3.0])
        self.assertEqual(c5.getDelays(3).tolist(), [1.0, 2.0, 3.0]) 
        #c6 = common.Connector(weights=0.5, delays=0.0, check_connections=True)
        #self.assertRaises(AssertionError, c6.getDelays, 3)
    
    def test_connect(self):
        c = common.Connector(delays=0.5)
        self.assertRaises(NotImplementedError, c.connect, 'foo')

class SynapticPlasticityTest(unittest.TestCase):
    
    def test_describe(self):
        s = common.SynapseDynamics()
        assert isinstance(s.describe(), basestring)
        assert isinstance(s.describe(template=None), dict)
       
    def test_stubs(self):
        self.assertRaises(NotImplementedError, common.ShortTermPlasticityMechanism)
        self.assertRaises(NotImplementedError, common.TsodyksMarkramMechanism)
        self.assertRaises(NotImplementedError, common.STDPWeightDependence)
        self.assertRaises(NotImplementedError, common.STDPTimingDependence)
        self.assertRaises(NotImplementedError, common.AdditiveWeightDependence)
        self.assertRaises(NotImplementedError, common.MultiplicativeWeightDependence)
        self.assertRaises(NotImplementedError, common.AdditivePotentiationMultiplicativeDepression)
        self.assertRaises(NotImplementedError, common.GutigWeightDependence)
        self.assertRaises(NotImplementedError, common.SpikePairRule)
                
# ==============================================================================
if __name__ == "__main__":
    unittest.main()