"""
Unit tests for pyNN/nest1.py.
$Id:nesttests.py 5 2007-04-16 15:01:24Z davison $
"""

import pyNN.nest1 as nest
import pyNN.common as common
import pyNN.random as random
import unittest
import numpy
import os

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def setUp(self):
        nest.setup()
    
    def tearDown(self):
        pass
    
    def testCreateStandardCell(self):
        """create(): First cell created should have GID==1"""
        ifcell = nest.create(nest.IF_curr_alpha)
        assert ifcell == 1, 'Failed to create standard cell'
        
    def testCreateStandardCells(self):
        """create(): Creating multiple cells should return a list of GIDs"""
        ifcell = nest.create(nest.IF_curr_alpha,n=10)
        assert ifcell == range(1,11), 'Failed to create 10 standard cells'
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        self.assertRaises(AssertionError, nest.create, nest.IF_curr_alpha, n=-1)
       
    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with getDict()"""
        ifcell = nest.create(nest.IF_curr_alpha,{'tau_syn_E':3.141592654})
        ifcell_params = nest.pynest.getDict([ifcell])
        assert ifcell_params[0]['TauSynE'] == 3.141592654
 
    def testCreateNESTCell(self):
        """create(): First cell created should have GID==1"""
        ifcell = nest.create('iaf_neuron')
        assert ifcell == 1, 'Failed to create NEST-specific cell'
    
    def testCreateNonExistentCell(self):
        """create(): Trying to create a cell type which is not a standard cell or
        a NEST cell should raise a SLIError."""
        self.assertRaises(nest.pynest.SLIError, nest.create, 'qwerty')
    
    #def testCreateWithInvalidParameter(self):
    #    """create(): Creating a cell with an invalid parameter should raise an Exception."""
    #    self.assertRaises(common.InvalidParameterError, nest.create, 'IF_curr_alpha', {'tau_foo':3.141592654})        
    
    #def __del__(self):
    #    nest.end()


# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        nest.setup(timestep=0.1,max_delay=5.0)
        self.postcells = nest.create(nest.IF_curr_alpha,n=3)
        self.precells = nest.create(nest.SpikeSourcePoisson,n=5)
        
    def testConnectTwoCells(self):
        """connect(): The first connection created should have id 0."""
        conn = nest.connect(self.precells[0],self.postcells[0])
        assert conn == 0, 'Error creating connection'
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        conn_id = nest.connect(self.precells[0],self.postcells[0],weight=0.1234)
        weight = nest.pynest.getWeight(nest.pynest.getAddress(self.precells[0]),conn_id)
        assert weight == 0.1234*1000, "Weight set does not match weight retrieved." # note that pyNN.nest uses nA for weights, whereas NEST uses pA

    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        conn_id = nest.connect(self.precells[0],self.postcells[0],delay=4.321)
        delay = nest.pynest.getDelay(nest.pynest.getAddress(self.precells[0]),conn_id)
        assert delay == 4.3, "Delay set does not match delay retrieved." # Note that delays are only stored to the precision of the timestep.

    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n, each element being the target port."""
        connlist = nest.connect(self.precells,self.postcells[0])
        assert connlist == [0]*len(self.precells)
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        connlist = nest.connect(self.precells[0],self.postcells)
        assert connlist == [0,1,2]
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        connlist = nest.connect(self.precells,self.postcells)
        assert connlist == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        connlist = nest.connect(self.precells,self.postcells,p=0.5)
        assert 0 < len(connlist) < len(self.precells)*len(self.postcells), 'Number of connections is %d: this is very unlikely (although possible).' % len(connlist)
    
    def testConnectNonExistentPreCell(self):
        """connect(): Connecting from non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, nest.connect, 12345, self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, nest.connect, self.precells[0], 45678)
        
    def testDelayTooSmall(self):
        """connect(): Setting a delay smaller than min_delay should raise an Exception.""" 
        self.assertRaises(common.ConnectionError, nest.connect, self.precells[0], self.postcells[0], delay=0.0)
           
    def testDelayTooLarge(self):
        """connect(): Setting a delay larger than max_delay should raise an Exception.""" 
        self.assertRaises(common.ConnectionError, nest.connect, self.precells[0], self.postcells[0], delay=1000.0)

    #def __del__(self):
    #    nest.end()

# ==============================================================================        
class SetValueTest(unittest.TestCase): pass # to write later

# ==============================================================================
class RecordSpikesTest(unittest.TestCase):
    
    def setUp(self):
        nest.setup()
        self.ifcell = nest.create(nest.IF_curr_alpha,{'i_offset':1.0})
    
    def testRecordSpikes(self):
        """record(): Just check no errors are raised."""
        nest.record(self.ifcell, 'test_record.tmp')
        nest.run(100.0)
        
# ==============================================================================
class RecordVTest(unittest.TestCase): pass # to write later

# ==============================================================================    
class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        nest.setup()
        nest.Population.nPop = 0
        
    def testSimpleInit(self):
        """Population.__init__(): should return a numpy array and give a default label."""
        net = nest.Population((3,3), nest.IF_curr_alpha)
        # shouldn't really have two assertions in one test but I'm lazy
        assert net.label == 'population0'                  
        assert numpy.equal(net.cell, numpy.array([[1,2,3],[4,5,6],[7,8,9]])).all() 
        
    def testInitWithParams(self):
        """Population.__init__(): Parameters set on creation should be the same as retrieved with getDict()"""
        net = nest.Population((3,3),nest.IF_curr_alpha,{'tau_syn_E':3.141592654})
        ifcell_params = nest.pynest.getDict([net.cell[0,0]])
        assert ifcell_params[0]['TauSynE'] == 3.141592654
    
    def testInitWithLabel(self):
        """Population.__init__(): A label set on initialisation should be retrievable with the Population.label attribute."""
        net = nest.Population((3,3),nest.IF_curr_alpha,label='iurghiushrg')
        assert net.label == 'iurghiushrg'
    
#    def testInvalidCellType(self):
#        """Population.__init__(): Trying to create a cell type which is not a method of StandardCells #should raise an AttributeError."""
#        self.assertRaises(AttributeError, nest.Population, (3,3), 'qwerty')
    
    #def __del__(self):
    #    nest.end()
    
    def testInitWithNonStandardModel(self):
        """Population.__init__(): should return a numpy array and give a default label."""
        net = nest.Population((3,3), 'iaf_neuron')
        assert net.label == 'population0'                  
        assert numpy.equal(net.cell, numpy.array([[1,2,3],[4,5,6],[7,8,9]])).all() 

# ==============================================================================
class PopulationIndexTest(unittest.TestCase):
    """Tests of the Population class indexing."""
    
    def setUp(self):
        nest.setup()
        nest.Population.nPop = 0
        self.net1 = nest.Population((10,),nest.IF_curr_alpha)
        self.net2 = nest.Population((2,4,3),nest.IF_curr_exp)
        self.net3 = nest.Population((2,2,1),nest.SpikeSourceArray)
        self.net4 = nest.Population((1,2,1),nest.SpikeSourceArray)
        self.net5 = nest.Population((3,3),nest.IF_cond_exp)
    
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
        nest.setup()
        nest.Population.nPop = 0
        self.net1 = nest.Population((10,),nest.IF_curr_alpha)
        self.net2 = nest.Population((2,4,3),nest.IF_curr_exp)
        self.net3 = nest.Population((2,2,1),nest.SpikeSourceArray)
        self.net4 = nest.Population((1,2,1),nest.SpikeSourceArray)
        self.net5 = nest.Population((3,3),nest.IF_cond_exp)
        
    def testIter(self):
        """This needs more thought for the distributed case."""
        for net in self.net1, self.net2:
            ids = [i for i in net]
            self.assertEqual(ids, net.cell.flatten().tolist())
            self.assert_(isinstance(ids[0], nest.ID))
            
    def testAddressIter(self):
        for net in self.net1, self.net2:
            for id,addr in zip(net.ids(),net.addresses()):
                self.assertEqual(id, net[addr])
                self.assertEqual(addr, net.locate(id))

# ==============================================================================
class PopulationSetTest(unittest.TestCase):
    """Tests of the set(), tset() and rset() methods of the Population class."""

    #def __del__(self):
    #    nest.end()
        
    def setUp(self):
        nest.setup()
        nest.Population.nPop = 0
        self.net = nest.Population((3,3),nest.IF_curr_alpha)
        self.net2 = nest.Population((5,),'iaf_neuron')
    
    def testSetFromDict(self):
        """Parameters set in a dict should all be retrievable using pynest.getDict()"""
        self.net.set({'tau_m':43.21, 'cm':0.987})
        assert nest.pynest.getDict([self.net.cell[0,0]])[0]['Tau'] == 43.21
        assert nest.pynest.getDict([self.net.cell[0,0]])[0]['C'] == 987.0 # pF
    
    def testSetFromPair(self):
        """A parameter set as a string,value pair should be retrievable using pynest.getDict()"""
        self.net.set('tau_m',12.34)
        assert nest.pynest.getDict([self.net.cell[0,0]])[0]['Tau'] == 12.34
    
    def testSetInvalidFromPair(self):
        """Trying to set an invalid value for a parameter should raise an exception."""
        self.assertRaises(common.InvalidParameterValueError, self.net.set, 'tau_m', [])
    
    def testSetInvalidFromDict(self):
        """When any of the parameters in a dict have invalid values, then an exception should be raised.
           There is no guarantee that the valid parameters will be set."""
        self.assertRaises(common.InvalidParameterValueError, self.net.set, {'v_thresh':'hello','tau_m':56.78})
    
    def testSetNonexistentFromPair(self):
        """Trying to set a nonexistent parameter should raise an exception."""
        self.assertRaises(common.NonExistentParameterError, self.net.set, 'tau_foo', 10.0)
    
    def testSetNonexistentFromDict(self):
        """When some of the parameters in a dict are inexistent, an exception should be raised.
           There is no guarantee that the existing parameters will be set."""
        self.assertRaises(common.NonExistentParameterError, self.net.set, {'tau_foo': 10.0, 'tau_m': 21.0})
    
    def testSetWithNonStandardModel(self):
        """Parameters set in a dict should all be retrievable using pynest.getDict()"""
        self.net2.set({'Tau':43.21})
        assert nest.pynest.getDict([self.net2.cell[0]])[0]['Tau'] == 43.21
    
    def testTSet(self):
        """The valueArray passed should be retrievable using population.get() on all nodes."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.net.tset('cm', array_in)
        array_out = self.net.get('cm', as_array=True).reshape((3,3))
        self.assert_((array_in == array_out).all(), "%s != %s" % (array_in, array_out))
    
    def testTSetArrayUnchanged(self):
        array_in1 = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        array_in2 = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.assert_((array_in1==array_in2).all())
        self.net.tset('cm', array_in1)
        self.assert_((array_in1==array_in2).all())
    
    def testTSetInvalidDimensions(self):
        """If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.assertRaises(common.InvalidDimensionsError, self.net.tset, 'i_offset', array_in)
    
    def testTSetInvalidValues(self):
        """If some of the values in the valueArray are invalid, should raise an exception."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,'apples']])
        self.assertRaises(common.InvalidParameterValueError, self.net.tset, 'i_offset', array_in)
        
    def testRSetNumpy(self):
        """Population.rset(): with numpy rng."""
        rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.net.rset('cm',rd1)
        output_values = self.net.get('cm', as_array=True)
        input_values = rd2.next(9)
        for i in range(9):
            self.assertAlmostEqual(input_values[i],output_values[i],places=5)
            
# ==============================================================================
class PopulationCallTest(unittest.TestCase): # to write later
    """Tests of the call() and tcall() methods of the Population class."""
    pass

# ==============================================================================
class PopulationRecordTest(unittest.TestCase): # to write later
    """Tests of the record(), record_v(), printSpikes(), print_v() and
       meanSpikeCount() methods of the Population class."""
    def setUp(self):
        nest.setup()
        nest.Population.nPop = 0
        self.pop1 = nest.Population((3,3), nest.SpikeSourcePoisson,{'rate': 20.})
        self.pop2 = nest.Population((3,3), nest.IF_curr_alpha)

    def testRecordAll(self):
        """Population.record(): not a full test, just checking there are no Exceptions raised."""
        self.pop1.record()
        
    def testRecordInt(self):
        """Population.record(n): not a full test, just checking there are no Exceptions raised."""
        # Partial record        
        self.pop1.record(5)
        
    def testRecordWithRNG(self):
        """Population.record(n,rng): not a full test, just checking there are no Exceptions raised."""
        self.pop1.record(5,random.NumpyRNG())
        
    def testRecordList(self):
        """Population.record(list): not a full test, just checking there are no Exceptions raised."""
        # Selected list record
        record_list = []
        for i in range(0,2):
            record_list.append(self.pop1[i,1])
        self.pop1.record(record_list) 
   
    def testSpikeRecording(self):
        # We test the mean spike count by checking if the rate of the poissonian sources are
        # close to 20 Hz. Then we also test how the spikes are saved
        self.pop1.record()
        simtime = 1000.0
        nest.run(simtime)
        self.pop1.printSpikes("temp_nest.ras")
        rate = self.pop1.meanSpikeCount()*1000.0/simtime
        assert (20*0.8 < rate) and (rate < 20*1.2), rate
        
    def testPotentialRecording(self):
        """Population.record_v() and Population.print_v(): not a full test, just checking 
        # there are no Exceptions raised."""
        rng = random.NumpyRNG(123)
        v_reset  = -65.0
        v_thresh = -50.0
        uniformDistr = random.RandomDistribution(rng=rng,distribution='uniform',parameters=[v_reset,v_thresh])
        self.pop2.randomInit(uniformDistr)
        self.pop2.record_v([self.pop2[0,0], self.pop2[1,1]])
        simtime = 10
        nest.run(simtime)
        self.pop2.print_v("temp_nest.v")

# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Projection class."""
        
    def setUp(self):
        nest.setup(max_delay=0.5)
        nest.Population.nPop = 0
        self.target33 = nest.Population((3,3),nest.IF_curr_alpha)
        self.target6  = nest.Population((6,),nest.IF_curr_alpha)
        self.source5  = nest.Population((5,),nest.SpikeSourcePoisson)
        self.source22 = nest.Population((2,2),nest.SpikeSourcePoisson)
        self.source33 = nest.Population((3,3),nest.SpikeSourcePoisson)

    def testAllToAll(self):
        """For all connections created with "allToAll" it should be possible to obtain the weight using pynest.getWeight()"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = nest.Projection(srcP, tgtP, "allToAll")
                prj2 = nest.Projection(srcP, tgtP, nest.AllToAllConnector())
                for prj in prj1,prj2:
                    assert len(prj._sources) == len(prj._targets)
                    weights = []
                    for src,tgt in prj.connections():
                        weights.append(nest.pynest.getWeight(src,tgt))
                    assert weights == [0.0]*len(prj._sources) # default weight is zero
    
    def testOneToOne(self):
        """For all connections created with "OneToOne" it should be possible to obtain the weight using pyneuron.getWeight()"""
        prj1 = nest.Projection(self.source33, self.target33, 'oneToOne')
        prj2 = nest.Projection(self.source33, self.target33, nest.OneToOneConnector())
        assert len(prj1) == self.source33.size
        assert len(prj2) == self.source33.size
        
    def testDistantDependentProbability(self):
        """For all connections created with "distanceDependentProbability"..."""
        # Test should be improved..."

        for rngclass in (nest.NumpyRNG, nest.NativeRNG):
            for expr in ('exp(-d)', 'd < 0.5'):
                prj1 = nest.Projection(self.source33, self.target33,
                                         'distanceDependentProbability',
                                         {'d_expression' : expr},rng=rngclass(12345))
                prj2 = nest.Projection(self.source33, self.target33,
                                         nest.DistanceDependentProbabilityConnector(d_expression=expr),
                                         rng=rngclass(12345))
                assert (0 < len(prj1) < len(self.source33)*len(self.target33)) \
                   and (0 < len(prj2) < len(self.source33)*len(self.target33))
                if rngclass == nest.NumpyRNG:
                    assert prj1._sources == prj2._sources, "%s %s" % (rngclass, expr)
                    assert prj1._targets == prj2._targets, "%s %s" % (rngclass, expr)

    def testFixedProbability(self):
        """For all connections created with "fixedProbability" it should be possible to obtain the weight using pynest.getWeight()"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = nest.Projection(srcP, tgtP, "fixedProbability", 0.5)
                prj2 = nest.Projection(srcP, tgtP, nest.FixedProbabilityConnector(0.5))
                for prj in prj1, prj2:
                    assert len(prj._sources) == len(prj._targets)
                    weights = []
                    for src, tgt in prj.connections():
                        weights.append(nest.pynest.getWeight(src,tgt))
                    assert weights == [0.0]*len(prj._sources), weights
                    
    def testSaveAndLoad(self):
        prj1 = nest.Projection(self.source22, self.target33, 'allToAll')
        prj1.setDelays(0.2)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp")
        connector = nest.FromFileConnector("connections.tmp")
        prj2 = nest.Projection(self.source22, self.target33, connector)
        w1 = []; w2 = []; d1 = []; d2 = [];
        # For a connections scheme saved and reloaded, we test if the connections, their weights and their delays
        # are equal.
        for src,tgt in prj1.connections():
            w1.append(nest.pynest.getWeight(src,tgt))
            d1.append(nest.pynest.getDelay(src,tgt))
        for src,tgt in prj2.connections():
            w2.append(nest.pynest.getWeight(src,tgt))
            d2.append(nest.pynest.getDelay(src,tgt))
        assert (w1 == w2) and (d1 == d2)

class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), randomizeWeights() and
    randomizeDelays() methods of the Projection class."""

    def setUp(self):
        nest.setup(max_delay=0.5)
        nest.Population.nPop = 0
        self.target33 = nest.Population((3,3),nest.IF_curr_alpha)
        self.target6  = nest.Population((6,),nest.IF_curr_alpha)
        self.source5  = nest.Population((5,),nest.SpikeSourcePoisson)
        self.source22 = nest.Population((2,2),nest.SpikeSourcePoisson)
        self.prjlist = []
        self.distrib_Numpy = random.RandomDistribution(rng=random.NumpyRNG(12345),distribution='uniform',parameters=(0.1,0.5)) 
        for tgtP in [self.target6, self.target33]:
            for srcP in [self.source5, self.source22]:
                for method in ('allToAll', 'fixedProbability'):
                    self.prjlist.append( nest.Projection(srcP,tgtP,method,{'p_connect':0.5}) )

    def testSetWeightsToSingleValue(self):
        """Weights set using setWeights() should be retrievable with pynest.getWeight()"""
        for prj in self.prjlist:
            prj.setWeights(1.234)
            for src, tgt in prj.connections():
                assert nest.pynest.getWeight(src,tgt) == 1234.0 # note the difference in units between pyNN and NEST

    #def testSetAndGetID(self):
        # Small test to see if the ID class is working
        #self.target33[0,2].set({'tau_m' : 15.1})
        #assert (self.target33[0,2].get('tau_m') == 15.1)
        
    def testrandomizeWeights(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = nest.Projection(self.source5, self.target33, 'allToAll')
        prj1.randomizeWeights(self.distrib_Numpy)
        w1 = []; w2 = [];
        for src,tgt in prj1.connections():
            w1.append(nest.pynest.getWeight(src,tgt))
        prj1.randomizeWeights(self.distrib_Numpy)        
        for src, tgt in prj1.connections():
            w2.append(nest.pynest.getWeight(src,tgt)) 
        self.assertNotEqual(w1,w2)
        
    def testrandomizeDelays(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = nest.Projection(self.source5, self.target33, 'allToAll')
        prj1.randomizeDelays(self.distrib_Numpy)
        d1 = []; d2 = [];
        for src,tgt in prj1.connections():
            d1.append(nest.pynest.getDelay(src,tgt))
        prj1.randomizeDelays(self.distrib_Numpy)        
        for src, tgt in prj1.connections():
            d2.append(nest.pynest.getDelay(src,tgt)) 
        self.assertNotEqual(d1,d2)

class IDTest(unittest.TestCase):
    """Tests of the ID class."""
    
    def setUp(self):
        nest.setup(max_delay=0.5)
        nest.Population.nPop = 0
        self.pop1 = nest.Population((5,),nest.IF_curr_alpha,{'tau_m':10.0})
        self.pop2 = nest.Population((5,4),nest.IF_curr_exp,{'v_reset':-60.0})
    
    def testIDSetAndGet(self):
        self.pop1[3].tau_m = 20.0
        self.pop2[3,2].v_reset = -70.0
        ifcell_params = nest.pynest.getDict([self.pop1[3]])[0]
        self.assertEqual(20.0, ifcell_params['Tau'])
        self.assertEqual(20.0, self.pop1[3].tau_m)
        self.assertEqual(10.0, self.pop1[0].tau_m)
        self.assertEqual(-70.0, self.pop2[3,2].v_reset)
        self.assertEqual(-60.0, self.pop2[0,0].v_reset)

    def testGetCellClass(self):
        self.assertEqual(nest.IF_curr_alpha, self.pop1[0].cellclass)
        self.assertEqual(nest.IF_curr_exp, self.pop2[4,3].cellclass)
        
    def testSetAndGetPosition(self):
        self.assert_((self.pop2[0,2].position == (0.0,2.0,0.0)).all())
        new_pos = (0.5,1.5,0.0)
        self.pop2[0,2].position = new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())
        new_pos = (-0.6,3.5,-100.0) # check that position is set-by-value from new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())


if __name__ == "__main__":
    unittest.main()
    
