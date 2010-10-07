"""
Unit tests for the common, cells, synapses, connectors modules
$Id$
"""

import sys
import unittest
import numpy
import os
from math import sqrt
from pyNN import common, random, cells, synapses, connectors, standardmodels, errors, space

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) <= threshold).all()

class MockSimulator(object):
    class MockState(object):
        min_delay = 0.1
    state = MockState()
common.simulator = MockSimulator

# ==============================================================================
class ExceptionsTest(unittest.TestCase):
    
    def test_NonExistentParameterError_withStandardModel(self):
        try:
            raise errors.NonExistentParameterError("foo", cells.IF_cond_alpha)
        except errors.NonExistentParameterError, err:
            self.assertEqual(err.model_name, 'IF_cond_alpha')
            self.assertEqual(err.parameter_name, 'foo')
            self.assertEqual(err.valid_parameter_names,
                             ['cm', 'e_rev_E', 'e_rev_I', 'i_offset', 'tau_m',
                              'tau_refrac', 'tau_syn_E', 'tau_syn_I',
                              'v_reset', 'v_rest', 'v_thresh'])
            assert len(str(err)) > 0
    
    def test_NonExistentParameterError_withStringModel(self):
        try:
            raise errors.NonExistentParameterError("foo", 'iaf_neuron')
        except errors.NonExistentParameterError, err:
            self.assertEqual(err.model_name, 'iaf_neuron')
            self.assertEqual(err.parameter_name, 'foo')
            self.assertEqual(err.valid_parameter_names, ['unknown'])
            assert len(str(err)) > 0
    
    def test_NonExistentParameterError_withInvalidModel(self):
        # model must be a string or a class
        self.assertRaises(Exception, errors.NonExistentParameterError, "foo", [])
    

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

class SpaceTest(unittest.TestCase):
    
    def setUp(self):
        N = numpy.array
        self.A = N([0.0, 0.0, 0.0])
        self.B = N([1.0, 1.0, 1.0])
        self.C = N([-1.0, -1.0, -1.0])
        self.D = N([2.0, 3.0, 4.0])
        self.ABCD = N([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0],
                       [-1.0, -1.0, -1.0],
                       [2.0, 3.0, 4.0]]).T

    def assertArraysEqual(self, A, B):
        self.assert_((A==B).all(), "%s != %s" % (A,B))
    
    def test_infinite_space_with_3D_distances(self):
        s = space.Space()
        self.assertEqual(s.distances(self.A, self.B), sqrt(3))
        self.assertEqual(s.distances(self.C, self.B), sqrt(12))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               numpy.array([0.0, sqrt(3), sqrt(3), sqrt(29)]))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               s.distances(self.ABCD, self.A).T)
    
    def test_infinite_space_with_collapsed_axes(self):
        s_x = space.Space(axes='x')
        s_xy = space.Space(axes='xy')
        s_yz = space.Space(axes='yz')
        self.assertEqual(s_x.distances(self.A, self.B), 1.0)
        self.assertEqual(s_xy.distances(self.A, self.B), sqrt(2))
        self.assertEqual(s_x.distances(self.A, self.D), 2.0)
        self.assertEqual(s_xy.distances(self.A, self.D), sqrt(13))
        self.assertEqual(s_yz.distances(self.A, self.D), sqrt(25))
        self.assertArraysEqual(s_yz.distances(self.D, self.ABCD),
                               numpy.array([sqrt(25), sqrt(13), sqrt(41), sqrt(0)]))
    
    def test_infinite_space_with_scale_and_offset(self):
        s = space.Space(scale_factor=2.0, offset=1.0)
        self.assertEqual(s.distances(self.A, self.B), sqrt(48))
        self.assertEqual(s.distances(self.B, self.A), sqrt(3))
        self.assertEqual(s.distances(self.C, self.B), sqrt(75))
        self.assertEqual(s.distances(self.B, self.C), sqrt(3))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               numpy.array([sqrt(12), sqrt(48), sqrt(0), sqrt(200)]))
    
    def test_cylindrical_space(self):
        s = space.Space(periodic_boundaries=((-1.0, 4.0), (-1.0, 4.0), (-1.0, 4.0)))
        self.assertEqual(s.distances(self.A, self.B), sqrt(3))
        self.assertEqual(s.distances(self.A, self.D), sqrt(4+4+1))
        self.assertEqual(s.distances(self.C, self.D), sqrt(4+1+0))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               numpy.array([0.0, sqrt(3), sqrt(3), sqrt(4+4+1)]))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               s.distances(self.ABCD, self.A).T)
        self.assertArraysEqual(s.distances(self.C, self.ABCD),
                               numpy.array([sqrt(3), sqrt(4+4+4), 0.0, sqrt(4+1+0)]))
    
    
class StandardModelTest(unittest.TestCase):
    
    def testCheckParameters(self):
        self.assertRaises(errors.InvalidParameterValueError, cells.SpikeSourceArray, {'spike_times': 0.0})
        self.assertRaises(errors.InvalidParameterValueError, cells.SpikeSourceInhGamma, {'a': 'foo'})
        self.assertRaises(ValueError, cells.SpikeSourceArray, {'spike_times': 'foo'})
        self.assertRaises(errors.NonExistentParameterError, cells.IF_cond_exp, {'foo': 'bar'})
        
    def testTranslate(self):
        class FakeCellType(standardmodels.StandardCellType):
            default_parameters = {'foo': 3, 'other_parameter': 5}
        class SimFakeCellType1(FakeCellType):
            translations = standardmodels.build_translations(('foo', 'bar', 'foo*non_existent_parameter', 'bar/non_existent_parameter'))
        class SimFakeCellType2(FakeCellType):
            translations = standardmodels.build_translations(
                ('foo', 'bar', 'foo*other_parameter', 'bar/non_existent_parameter'),
                ('other_parameter', 'translated_other_parameter'),    
            )
        self.assertRaises(NameError, SimFakeCellType1, {'foo': 4})
        cell_type = SimFakeCellType2({'foo': 4, 'other_parameter': 5})
        self.assertRaises(NameError, cell_type.reverse_translate, {'bar': 20})
        assert isinstance(cell_type.describe(), basestring) # this belongs in a separate test
        
    def testCreatingNonAvailableModel(self):
        self.assertRaises(NotImplementedError, standardmodels.ModelNotAvailable)
                              

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
               
        
class ConnectorTest(unittest.TestCase):

    def test_get_weights(self):
        c1 = connectors.Connector(delays=0.5, weights=0.5)
        self.assertEqual(c1.get_weights(3).tolist(), [0.5,0.5,0.5])
        c2 = connectors.Connector(delays=0.5, weights="foo")
        self.assertRaises(ValueError, c2.get_weights, 3)
        class A(object): pass
        c3 = connectors.Connector(delays=0.5, weights=A())
        self.assertRaises(Exception,c3.get_weights, 3)
        rd = random.RandomDistribution('gamma', [0.5,0.5])
        c4 = connectors.Connector(delays=0.5, weights=rd)
        w = c4.get_weights(3)
        self.assertEqual(len(w), 3)
        self.assertNotEqual(w[0], w[1])
        
    def test_get_delays(self):
        c1 = connectors.Connector(delays=0.5, weights=0.5)
        self.assertEqual(c1.get_delays(3).tolist(), [0.5,0.5,0.5])
        c2 = connectors.Connector(weights=0.5, delays="foo")
        self.assertRaises(ValueError, c2.get_delays, 3)
        class A(object): pass
        c3 = connectors.Connector(weights=0.5, delays=A())
        self.assertRaises(Exception,c3.get_delays, 3)
        rd = random.RandomDistribution('gamma', [0.5,0.5])
        c4 = connectors.Connector(weights=0.5, delays=rd)
        d = c4.get_delays(3)
        self.assertEqual(len(d), 3)
        self.assertNotEqual(d[0], d[1])
        c5 = connectors.Connector(weights=0.5, delays=[1.0, 2.0, 3.0])
        self.assertEqual(c5.get_delays(3).tolist(), [1.0, 2.0, 3.0]) 
    
    def test_connect(self):
        c = connectors.Connector(delays=0.5)
        self.assertRaises(NotImplementedError, c.connect, 'foo')

class SynapticPlasticityTest(unittest.TestCase):
    
    def test_describe(self):
        s = standardmodels.SynapseDynamics()
        assert isinstance(s.describe(), basestring)
        assert isinstance(s.describe(template=None), dict)
       
    def test_stubs(self):
        self.assertRaises(NotImplementedError, standardmodels.ShortTermPlasticityMechanism)
        self.assertRaises(NotImplementedError, synapses.TsodyksMarkramMechanism)
        self.assertRaises(NotImplementedError, standardmodels.STDPWeightDependence)
        self.assertRaises(NotImplementedError, standardmodels.STDPTimingDependence)
        self.assertRaises(NotImplementedError, synapses.AdditiveWeightDependence)
        self.assertRaises(NotImplementedError, synapses.MultiplicativeWeightDependence)
        self.assertRaises(NotImplementedError, synapses.AdditivePotentiationMultiplicativeDepression)
        self.assertRaises(NotImplementedError, synapses.GutigWeightDependence)
        self.assertRaises(NotImplementedError, synapses.SpikePairRule)


# ==============================================================================
if __name__ == "__main__":
    unittest.main()