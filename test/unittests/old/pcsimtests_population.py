"""
Unit tests for pyNN.pcsim package

    Unit tests for verifying the correctness of the high level population based 
    part of the pyNN interface.

    Dejan Pecevski, March, 2007
        dejan@igi.tugraz.at  

"""

import pyNN.common as common
import pyNN.random as random
from pyNN.pcsim import *
from pypcsim import *
import unittest, sys, numpy
import numpy.random

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) < threshold).all()

def max_array_diff(a, b):
    return max(abs(a-b))

class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        setup()
        
    
    def tearDown(self):
        pass
        
    def testSimpleInit(self):
        """Population.__init__(): the cell list in hoc should have the same length as the population size."""
        popul = Population((3,3),IF_curr_alpha)
        self.assertEqual( len(popul), 9 )        
        
    
    def testInitWithParams(self):
        """Population.__init__(): Parameters set on creation should be the same as retrieved with HocToPy.get()"""
        popul = Population((3,3),IF_curr_alpha,{'tau_syn_E':3.141592654})
        tau_syn = simulator.net.object(popul.getObjectID(popul[2,2])).TauSynExc 
        self.assertAlmostEqual(tau_syn, 0.003141592654, places=5)
    
    def testInitWithLabel(self):
        """Population.__init__(): A label set on initialisation should be retrievable with the Population.label attribute."""
        popul = Population((3,3),IF_curr_alpha,label='iurghiushrg')
        assert popul.label == 'iurghiushrg'
    
#    def testInvalidCellType(self):
#        """Population.__init__(): Trying to create a cell type which is not a method of StandardCells should raise an AttributeError."""
#        self.assertRaises(AttributeError, neuron.Population, (3,3), 'qwerty', {})

    def testInitWithNonStandardModel(self):
        """Population.__init__(): the cell list in hoc should have the same length as the population size."""
        popul = Population((3,3),'LifNeuron',{'Rm':5e6,'Vthresh':-0.055})
        popul2 = Population((3,3),LifNeuron,{'Rm':5e6,'Vthresh':-0.055}) # pcsim allows also specification of non-standard models as types
        self.assertEqual(len(popul), 9)
        self.assertEqual(len(popul2), 9)

# ==============================================================================
class PopulationIndexTest(unittest.TestCase):
    """Tests of the Population class indexing."""
    
    def setUp(self):
        setup()
        Population.nPop = 0
        self.net1 = Population((10,),IF_curr_alpha)
        self.net2 = Population((2,4,3),IF_curr_exp)
        self.net3 = Population((2,2,1),SpikeSourceArray)
        self.net4 = Population((1,2,1),SpikeSourceArray)
        self.net5 = Population((3,3),IF_curr_exp)
    
    def testValidIndices(self):
        for i in range(10):
            self.assertEqual((i,),self.net1.locate(self.net1[i]))

    def testValidAddresses(self):
        for addr in ( (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2), (0,3,0), (0,3,1), (0,3,2),
                      (1,0,0), (1,0,1), (1,0,2), (1,1,0), (1,1,1), (1,1,2), (1,2,0), (1,2,1), (1,2,2), (1,3,0), (1,3,1), (1,3,2) ):
            id = self.net2[addr]
            self.assertEqual(addr, self.net2.locate(id))
        for addr in ( (0,0,0), (0,1,0), (1,0,0), (1,1,0) ):
            id = self.net3[addr]
            self.assertEqual(addr, self.net3.locate(id))
        for addr in ( (0,0,0), (0,1,0) ):
            id = self.net4[addr]
            self.assertEqual(addr, self.net4.locate(id))
        for addr in ( (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2) ):
            id = self.net5[addr]
            self.assertEqual(addr, self.net5.locate(id))

    def testInvalidIndices(self):
        self.assertRaises(IndexError, self.net1.__getitem__, (11,))
        
    def testInvalidIndexDimension(self):
        self.assertRaises(common.InvalidDimensionsError, self.net1.__getitem__, (10,2))

# ==============================================================================
class PopulationIteratorTest(unittest.TestCase):
    """Tests of the Population class iterators."""
    
    def setUp(self):
        setup()
        Population.nPop = 0
        self.net1 = Population((10,),IF_curr_alpha)
        self.net2 = Population((2,4,3),IF_curr_exp)
        self.net3 = Population((2,2,1),SpikeSourceArray)
        self.net4 = Population((1,2,1),SpikeSourceArray)
        self.net5 = Population((3,3),IF_curr_exp)
        
    def testIter(self):
        """This needs more thought for the distributed case."""
        for net in self.net1, self.net2:
            ids = [i for i in net]
            idVec = numpy.array(net.pcsim_population.idVector())
            idVec -= idVec[0]
            self.assertEqual(ids, idVec.tolist())
            self.assert_(isinstance(ids[0], ID))
            
    def testAddressIter(self):
        for net in self.net1, self.net2:
            for id,addr in zip(net.ids(),net.addresses()):
                self.assertEqual(id, net[addr])
                self.assertEqual(addr, net.locate(id))
                
 # ==============================================================================
class PopulationSetTest(unittest.TestCase):
         
    def setUp(self):
        setup()
        Population.nPop = 0
        self.popul1 = Population((3,3),IF_curr_alpha)
        self.popul2 = Population((5,),CbLifNeuron,{'Vinit':-0.070, 'Inoise':0.001})
    
    def testSetFromDict(self):
        """Population.set(): Parameters set in a dict should all be retrievable from PyPCSIM directly"""
        self.popul1.set({'tau_m':43.21})
        self.assertAlmostEqual( simulator.net.object(self.popul1.getObjectID(8)).taum, 0.04321, places = 5)
#     
    def testSetFromPair(self):
       """Population.set(): A parameter set as a string,value pair should be retrievable using PyPCSIM directly"""
       self.popul1.set('tau_m',12.34)
       self.popul1.set('v_init',-65.0)
       self.assertAlmostEqual( simulator.net.object(self.popul1.getObjectID(3)).taum, 0.01234, places=5)
       self.assertAlmostEqual( simulator.net.object(self.popul1.getObjectID(3)).Vinit, -0.065, places=5)
      
    
    def testSetInvalidFromPair(self):
        """Population.set(): Trying to set an invalid value for a parameter should raise an exception."""
        self.assertRaises(common.InvalidParameterValueError, self.popul1.set, 'tau_m', [])
    
    def testSetInvalidFromDict(self):
        """Population.set(): When any of the parameters in a dict have invalid values, then an exception should be raised.
           There is no guarantee that the valid parameters will be set."""
        self.assertRaises(common.InvalidParameterValueError, self.popul1.set, {'v_thresh':'hello','tau_m':56.78})
    
    def testSetNonexistentFromPair(self):
        """Population.set(): Trying to set a nonexistent parameter should raise an exception."""
        self.assertRaises(common.NonExistentParameterError, self.popul1.set, 'tau_foo', 10.0)
    
    def testSetNonexistentFromDict(self):
        """Population.set(): When some of the parameters in a dict are inexistent, an exception should be raised.
           There is no guarantee that the existing parameters will be set."""
        self.assertRaises(common.NonExistentParameterError, self.popul1.set, {'tau_foo': 10.0, 'tau_m': 21.0})
    
    def testSetWithNonStandardModel(self):
        """Population.set(): Parameters set in a dict should all be retrievable using PyPCSIM interface directly"""
        self.popul2.set({'Rm':4.5e6})
        self.assertAlmostEqual( simulator.net.object(self.popul2.getObjectID(3)).Rm , 4.5e6, places = 10)
        
    def testTSet(self):
        """Population.tset(): The valueArray passed should be retrievable using the PyPCSIM interface """
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.popul1.tset('i_offset', array_in)
        for i in 0,1,2:
            for j in 0,1,2:
                self.assertAlmostEqual( array_in[i,j], simulator.net.object(self.popul1.getObjectID(self.popul1[i,j])).Iinject*1e9 , places = 7 )
    
    def testTSetArrayUnchanged(self):
       array_in1 = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
       array_in2 = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
       self.assert_((array_in1==array_in2).all())
       self.popul1.tset('i_offset', array_in1)
       self.assert_((array_in1==array_in2).all())
    
    def testTSetInvalidDimensions(self):
        """Population.tset(): If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.assertRaises(common.InvalidDimensionsError, self.popul1.tset, 'i_offset', array_in)
    
    def testTSetInvalidValues(self):
        """Population.tset(): If some of the values in the valueArray are invalid, should raise an exception."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,'apples']])
        self.assertRaises(common.InvalidParameterValueError, self.popul1.tset, 'i_offset', array_in)
        """Population.rset(): with native rng. This is difficult to test, so for now just require that all values retrieved should be different. Later, could calculate distribution and assert that the difference between sample and theoretical distribution is less than some threshold."""
        
    def testRSetNative(self):
        self.popul1.rset('tau_m',
                      random.RandomDistribution(rng=NativeRNG(),
                                                distribution='Uniform',
                                                parameters=[10.0, 30.0]))
        self.assertNotEqual(simulator.net.object(self.popul1.getObjectID(3)).taum,
                            simulator.net.object(self.popul1.getObjectID(6)).taum)
        
    def testRSetNumpy(self):
         """Population.rset(): with numpy rng."""
         rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                          distribution='uniform',
                                          parameters=[0.9,1.1])
         rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                          distribution='uniform',
                                          parameters=[0.9,1.1])
         self.popul1.rset('cm',rd1)
         output_values = numpy.zeros((3,3),numpy.float)
         for i in 0,1,2:
             for j in 0,1,2:    
                 output_values[i,j] = 1e9*simulator.net.object(self.popul1.getObjectID(self.popul1[i,j])).Cm
         input_values = rd2.next(9)
         output_values = output_values.reshape((9,))
         for i in range(9):
             self.assertAlmostEqual(input_values[i],output_values[i],places=5)
         
    def testRSetNative2(self):
         """Population.rset(): with native rng."""
         rd1 = random.RandomDistribution(rng=NativeRNG(seed=98765),
                                          distribution='Uniform',
                                          parameters=[0.9,1.1])
         rd2 = random.RandomDistribution(rng=NativeRNG(seed=98765),
                                          distribution='Uniform',
                                          parameters=[0.9,1.1])
         self.popul1.rset('cm', rd1)
         output_values_1 = numpy.zeros((3,3),numpy.float)
         output_values_2 = numpy.zeros((3,3),numpy.float)
         for i in 0,1,2:
             for j in 0,1,2:
                 output_values_1[i,j] = simulator.net.object(self.popul1.getObjectID(self.popul1[i,j])).Cm
                 
         self.popul1.rset('cm', rd2)
         for i in 0,1,2:
             for j in 0,1,2:
                 output_values_2[i,j] = simulator.net.object(self.popul1.getObjectID(self.popul1[i,j])).Cm

         output_values_1 = output_values_1.reshape((9,))
         output_values_2 = output_values_2.reshape((9,))
         for i in range(9):
             self.assertAlmostEqual(output_values_1[i],output_values_2[i],places=5)    
        
# ==============================================================================
class PopulationCallTest(unittest.TestCase): # to write later
    """Tests of the _call() and _tcall() methods of the Population class."""
    pass

 # ==============================================================================
class PopulationRecordTest(unittest.TestCase): # to write later
    """Tests of the record(), record_v(), printSpikes(), print_v() and
       meanSpikeCount() methods of the Population class."""
    
    def setUp(self):
        Population.nPop = 0
        self.popul = Population((3,3),IF_curr_alpha)
        
    def tearDown(self):         
        end()
        
    def testRecordAll(self):
        """Population.record(): not a full test, just checking there are no Exceptions raised."""
        self.popul.record()
        
    def testRecordInt(self):
        """Population.record(n): not a full test, just checking there are no Exceptions raised."""
        self.popul.record(5)
        
    def testRecordWithRNG(self):
        """Population.record(n,rng): not a full test, just checking there are no Exceptions raised."""
        # self.popul.record(5,random.NumpyRNG())
        
    def testRecordList(self):
        """Population.record(list): not a full test, just checking there are no Exceptions raised."""
        self.popul.record([self.popul[(2,2)],self.popul[(1,2)],self.popul[(0,0)]])

 # ==============================================================================
class PopulationOtherTest(unittest.TestCase): # to write later
    """Tests of the randomInit() method of the Population class."""
    pass

# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Projection class."""
        
    def setUp(self):
        Population.nPop = 0
        # Projection.nProj = 0
        self.target33    = Population((3,3),IF_curr_alpha)
        self.target6     = Population((6,),IF_curr_alpha)
        self.source5     = Population((5,),SpikeSourcePoisson)
        self.source22    = Population((2,2),SpikeSourcePoisson)
        self.source33    = Population((3,3),SpikeSourcePoisson)
        self.expoisson33 = Population((3,3),SpikeSourcePoisson,{'rate': 100})
        
    def testAllToAll(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj = Projection(srcP, tgtP, AllToAllConnector())
                prj.setWeights(1.234)
                weights = []
                for i in range(len(prj)):
                    weights.append(simulator.net.object(prj.pcsim_projection[i]).W)
                for w in weights:
                    self.assertAlmostEqual(w,1.234*1e-9, places = 7)

    def testFixedProbability(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj = Projection(srcP, tgtP, FixedProbabilityConnector(0.5))
                assert (0 < len(prj) < len(srcP)*len(tgtP))
                
    def testOneToOne(self):
        prj = Projection(self.source33, self.target33, OneToOneConnector())
        assert len(prj) == self.source33.size
     
    def testDistantDependentProbability(self):
        """For all connections created with "distanceDependentProbability" ..."""
        # Test should be improved..."
        distrib_Numpy = random.RandomDistribution('uniform',(0,1),random.NumpyRNG(12345)) 
        distrib_Native= random.RandomDistribution('Uniform',(0,1),NativeRNG(12345)) 
        prj1 = Projection(self.source33, self.target33, DistanceDependentProbabilityConnector([ 0.1, 2]), distrib_Numpy)
        prj2 = Projection(self.source33, self.target33, DistanceDependentProbabilityConnector([ 0.1, 3]), distrib_Native)
        assert (0 < len(prj1) < len(self.source33)*len(self.target33)) and (0 < len(prj2) < len(self.source33)*len(self.target33))

    def testFromList(self):
        list_22_33 = [([0,0], [0,2], 0.2, 0.3),
                      ([1,0], [2,2], 0.3, 0.4),
                      ([1,0], [1,0], 0.4, 0.5)]
        prj1 = Projection(self.source22, self.target33, FromListConnector(list_22_33))
        assert len(prj1) == len(list_22_33)
        
    def testWithWeightArray(self):
        w = numpy.linspace(0.1,0.9,9)
        prj1 = Projection(self.source33, self.target33, OneToOneConnector(weights=w))
        assert len(prj1) == self.source33.size
        w_out = numpy.array(prj1.getWeights(format='list'))
        assert arrays_almost_equal(w_out, w, 1e-7), "Max difference is %g" % max_array_diff(w_out,w)
    
    def testWithDelayArray(self):
        d = numpy.linspace(1.1,1.9,9)
        prj1 = Projection(self.source33, self.target33, OneToOneConnector(delays=d))
        assert len(prj1) == self.source33.size
        d_out = numpy.array(prj1.getDelays(format='list'))
        assert arrays_almost_equal(d_out, d, 1e-7), "Max difference is %g" % max_array_diff(d_out,d)
    
    def testSaveAndLoad(self):
        prj1 = Projection(self.source33, self.target33, OneToOneConnector())
        prj1.setDelays(1)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp")
        prj2 = Projection(self.source33, self.target33, FromFileConnector("connections.tmp"))
        assert prj1.getWeights('list') == prj2.getWeights('list')
        assert prj1.getDelays('list') == prj2.getDelays('list')


class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), randomizeWeights() and
    randomizeDelays() methods of the Projection class."""

    def setUp(self):
        setup()
        self.target   = Population((3,3),IF_curr_alpha)
        self.target   = Population((3,3),IF_curr_alpha)
        self.source   = Population((3,3),SpikeSourcePoisson,{'rate': 100})
        self.distrib_Numpy = random.RandomDistribution('uniform',(0,1),random.NumpyRNG(12345)) 
        self.distrib_Native= random.RandomDistribution('Uniform',(0,1),NativeRNG(12345)) 
        
    def testsetWeights(self):
        prj1 = Projection(self.source, self.target, AllToAllConnector())
        prj1.setWeights(2.345)
        weights = []
        for i in range(len(prj1)):
            weights.append(simulator.net.object(prj1[i]).W)
        for w in weights:
            self.assertAlmostEqual(w, 2.345*1e-9)         
         
    def testrandomizeWeights(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = Projection(self.source, self.target, AllToAllConnector())
        prj2 = Projection(self.source, self.target, AllToAllConnector())
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        w1 = []; w2 = []; w3 = []; w4 = []
        for i in range(len(prj1)):
            w1.append(simulator.net.object(prj1[i]).W)
            w2.append(simulator.net.object(prj1[i]).W)
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        for i in range(len(prj1)):
            w3.append(simulator.net.object(prj1[i]).W)
            w4.append(simulator.net.object(prj1[i]).W)  
        self.assertNotEqual(w1,w3) and self.assertNotEqual(w2,w4)

        
    def testSetAndGetID(self):
        # Small test to see if the ID class is working
        # self.target[0,2].set({'tau_m' : 15.1})
        # assert (self.target[0,2].get('tau_m') == 15.1)
        pass
        
    def testSetAndGetPositionID(self):
        # Small test to see if the position of the ID class is working
        # self.target[0,2].setPosition((0.5,1.5))
        # assert (self.target[0,2].getPosition() == (0.5,1.5))
        pass
        

class IDTest(unittest.TestCase):
    """Tests of the ID class."""
    
    def setUp(self):
        setup(max_delay=0.5)
        self.pop1 = Population((5,),  IF_curr_alpha,{'tau_m':10.0})
        self.pop2 = Population((5,4), IF_curr_exp,{'v_reset':-60.0})

    def testIDSetAndGet(self):
        self.pop1[3].tau_m = 20.0
        self.pop2[3,2].v_reset = -70.0
        self.assertAlmostEqual( self.pop1.pcsim_population.object(self.pop1[3]).taum, 0.02, places = 5) 
        self.assertAlmostEqual(20.0, self.pop1[3].tau_m, places = 5)
        self.assertAlmostEqual(10.0, self.pop1[0].tau_m, places = 5)
        self.assertAlmostEqual(-70, self.pop2[3,2].v_reset, places = 5)
        self.assertAlmostEqual(-60, self.pop2[0,0].v_reset, places = 5)

    def testGetCellClass(self):
        self.assertEqual(IF_curr_alpha, self.pop1[0].cellclass)
        self.assertEqual(IF_curr_exp, self.pop2[4,3].cellclass)
        
    def testSetAndGetPosition(self):
        self.assert_((self.pop2[0,2].position == (0.0,2.0,0.0)).all())
        new_pos = (0.5,1.5,0.0)
        self.pop2[0,2].position = new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())
        new_pos = (-0.6,3.5,-100.0) # check that position is set-by-value from new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())

# ==============================================================================
if __name__ == "__main__":
    unittest.main()
