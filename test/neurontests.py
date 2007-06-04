"""
Unit tests for pyNN/neuron.py.
$Id$
"""

import pyNN.neuron as neuron
import pyNN.common as common
import pyNN.random as random
from pyNN.neuron import *
import unittest, sys, numpy
import numpy.random

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def tearDown(self):
        hoc_commands = ['objref cell%d' % i for i in neuron.gidlist]
        neuron.hoc_execute(hoc_commands,'=== CreationTest.tearDown() ===')
        neuron.gid = 0
        neuron.gidlist = []
    
    def testCreateStandardCell(self):
        """create(): First cell created should have index 0."""
        neuron.hoc_comment('=== CreationTest.testCreateStandardCell() ===')
        ifcell = neuron.create(neuron.IF_curr_alpha)
        assert ifcell == 0, 'Failed to create standard cell'
        
    def testCreateStandardCells(self):
        """create(): Creating multiple cells should return a list of integers"""
        neuron.hoc_comment('=== CreationTest.testCreateStandardCells ===')
        ifcells = neuron.create(neuron.IF_curr_alpha,n=10)
        assert ifcells == range(0,10), 'Failed to create 10 standard cells'
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        neuron.hoc_comment('=== CreationTest.testCreateStandardCellsWithNegative_n ===')
        self.assertRaises(AssertionError, neuron.create, neuron.IF_curr_alpha, n=-1)
       
    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with HocToPy.get()"""
        neuron.hoc_comment('=== CreationTest.testCreateStandardCellWithParams ===')
        ifcell = neuron.create(neuron.IF_curr_alpha,{'tau_syn':3.141592654})
        self.assertAlmostEqual(HocToPy.get('cell%d.esyn.tau' % ifcell, 'float'), 3.141592654, places=5)
    
    def testCreateNEURONCell(self):
        """create(): First cell created should have index 0."""
        neuron.hoc_comment('=== CreationTest.testCreateNEURONCell ===')
        ifcell = neuron.create('StandardIF',{'syn_type':'current','syn_shape':'exp'})
        assert ifcell == 0, 'Failed to create NEURON-specific cell'
    
#    def testCreateNonStandardCell(self):
#        """create(): Trying to create a cell type which is not a method of StandardCells should raise an AttributeError."""
#        self.assertRaises(AttributeError, neuron.create, 'qwerty')
    
    def testCreateWithInvalidParameter(self):
        """create(): Creating a cell with an invalid parameter should raise an Exception."""
        self.assertRaises(common.NonExistentParameterError, neuron.create, neuron.IF_curr_alpha, {'tau_foo':3.141592654})        


# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        neuron.hoc_comment("=== ConnectionTest.setUp() ===")
        self.postcells = neuron.create(neuron.IF_curr_alpha,n=3)
        self.precells = neuron.create(neuron.SpikeSourcePoisson,n=5)
        
    def tearDown(self):
        neuron.hoc_comment("=== ConnectionTest.tearDown() ===")
        hoc_commands = ['objref cell%d\n' % i for i in neuron.gidlist]
        hoc_commands += ['netconlist = new List()']
        neuron.hoc_execute(hoc_commands, '=== ConnectionTest.tearDown() ===')
        neuron.gid = 0
        neuron.ncid = 0
        neuron.gidlist = []
        
    def testConnectTwoCells(self):
        """connect(): The first connection created should have id 0."""
        neuron.hoc_comment("=== ConnectionTest.testConnectTwoCells ===")
        conn = neuron.connect(self.precells[0],self.postcells[0])
        assert conn == [0], 'Error creating connection'
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        neuron.hoc_comment("=== ConnectionTest.testConnectTwoCellsWithWeight() ===")
        conn_id = neuron.connect(self.precells[0],self.postcells[0],weight=0.1234)
        weight = HocToPy.get('netconlist.object(%d).weight' % conn_id[0], 'float')
        assert weight == 0.1234, "Weight set (0.1234) does not match weight retrieved (%s)" % weight
    
    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        conn_id = neuron.connect(self.precells[0],self.postcells[0],delay=4.321)
        delay = HocToPy.get('netconlist.object(%d).delay' % conn_id[0], 'float')
        assert delay == 4.321, "Delay set (4.321) does not match delay retrieved (%s)." % delay
    
    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n, each element being the id number of a netcon."""
        connlist = neuron.connect(self.precells,self.postcells[0])
        assert connlist == range(0,len(self.precells))
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        connlist = neuron.connect(self.precells[0],self.postcells)
        assert connlist == range(0,len(self.postcells))
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        connlist = neuron.connect(self.precells,self.postcells)
        assert connlist == range(0,len(self.postcells)*len(self.precells))
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        connlist = neuron.connect(self.precells,self.postcells,p=0.5)
        assert 0 < len(connlist) < len(self.precells)*len(self.postcells), 'Number of connections is %d: this is very unlikely (although possible).' % len(connlist)
    
    def testConnectNonExistentPreCell(self):
        """connect(): Connecting from non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, neuron.connect, 12345, self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, neuron.connect, self.precells[0], 'cell45678')
    
    def testInvalidSourceId(self):
        """connect(): sources must be integers."""
        self.precells.append('74367598')
        self.assertRaises(common.ConnectionError, neuron.connect, self.precells, self.postcells)
    
    def testInvalidTargetId(self):
        """connect(): targets must be integers."""
        self.postcells.append([])
        self.assertRaises(common.ConnectionError, neuron.connect, self.precells, self.postcells)

# ==============================================================================
class SetValueTest(unittest.TestCase):
    
    def setUp(self):
        self.cells = neuron.create(neuron.IF_curr_exp,n=10)
        
    def testSetFloat(self):
        neuron.hoc_comment("=== SetValueTest.testSetFloat() ===")
        neuron.set(self.cells,neuron.IF_curr_exp,'tau_m',35.7)
        for cell in self.cells:
            assert HocToPy.get('cell%d.tau_m' % cell, 'float') == 35.7
            
    #def testSetString(self):
    #    neuron.set(self.cells,neuron.IF_curr_exp,'param_name','string_value')
    ## note we don't currently have any models with string parameters, so
    ## this is all commented out
    #    for cell in self.cells:
    #        assert HocToPy.get('cell%d.param_name' % cell, 'string') == 'string_value'

    def testSetDict(self):
        neuron.set(self.cells,neuron.IF_curr_exp,{'tau_m':35.7,'tau_syn_E':5.432})
        for cell in self.cells:
            assert HocToPy.get('cell%d.tau_e' % cell, 'float') == 5.432
            assert HocToPy.get('cell%d.tau_m' % cell, 'float') == 35.7

    def testSetNonExistentParameter(self):
        # note that although syn_shape is added to the parameter dict when creating
        # an IF_curr_exp, it is not a valid parameter to be changed later.
        self.assertRaises(common.NonExistentParameterError,neuron.set,self.cells,neuron.IF_curr_exp,'syn_shape','alpha')

# ==============================================================================
class RecordTest(unittest.TestCase): pass # to do later

# ==============================================================================
class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
        
    def testSimpleInit(self):
        """Population.__init__(): the cell list in hoc should have the same length as the population size."""
        net = neuron.Population((3,3),neuron.IF_curr_alpha)
        assert HocToPy.get('%s.count()' % net.label, 'integer') == 9
    
    def testInitWithParams(self):
        """Population.__init__(): Parameters set on creation should be the same as retrieved with HocToPy.get()"""
        net = neuron.Population((3,3),neuron.IF_curr_alpha,{'tau_syn':3.141592654})
        tau_syn = HocToPy.get('%s.object(8).esyn.tau' % net.label)
        self.assertAlmostEqual(tau_syn, 3.141592654, places=5)
    
    def testInitWithLabel(self):
        """Population.__init__(): A label set on initialisation should be retrievable with the Population.label attribute."""
        net = neuron.Population((3,3),neuron.IF_curr_alpha,label='iurghiushrg')
        assert net.label == 'iurghiushrg'
    
#    def testInvalidCellType(self):
#        """Population.__init__(): Trying to create a cell type which is not a method of StandardCells should raise an AttributeError."""
#        self.assertRaises(AttributeError, neuron.Population, (3,3), 'qwerty', {})
        
    def testInitWithNonStandardModel(self):
        """Population.__init__(): the cell list in hoc should have the same length as the population size."""
        net = neuron.Population((3,3),'StandardIF',{'syn_type':'current','syn_shape':'exp'})
        assert HocToPy.get('%s.count()' % net.label, 'integer') == 9

# ==============================================================================
class PopulationIndexTest(unittest.TestCase):
    """Tests of the Population class indexing."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.net1 = neuron.Population((10,),neuron.IF_curr_alpha)
        self.net2 = neuron.Population((2,4,3),neuron.IF_curr_exp)
        self.net3 = neuron.Population((2,2,1),neuron.SpikeSourceArray)
        self.net4 = neuron.Population((1,2,1),neuron.SpikeSourceArray)
        self.net5 = neuron.Population((3,3),neuron.IF_cond_alpha)
    
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
        neuron.Population.nPop = 0
        self.net1 = neuron.Population((10,),neuron.IF_curr_alpha)
        self.net2 = neuron.Population((2,4,3),neuron.IF_curr_exp)
        self.net3 = neuron.Population((2,2,1),neuron.SpikeSourceArray)
        self.net4 = neuron.Population((1,2,1),neuron.SpikeSourceArray)
        self.net5 = neuron.Population((3,3),neuron.IF_cond_alpha)
        
    def testIter(self):
        """This needs more thought for the distributed case."""
        for net in self.net1, self.net2:
            ids = [i for i in net]
            self.assertEqual(ids, net.gidlist)
            self.assert_(isinstance(ids[0], neuron.ID))
            
    def testAddressIter(self):
        for net in self.net1, self.net2:
            for id,addr in zip(net.ids(),net.addresses()):
                self.assertEqual(id, net[addr])
                self.assertEqual(addr, net.locate(id))
            
    
# ==============================================================================
class PopulationSetTest(unittest.TestCase):
        
    def setUp(self):
        neuron.Population.nPop = 0
        self.net = neuron.Population((3,3),neuron.IF_curr_alpha)
        self.net2 = neuron.Population((5,),'StandardIF',{'syn_type':'current','syn_shape':'exp'})
    
    def testSetFromDict(self):
        """Population.set(): Parameters set in a dict should all be retrievable using HocToPy.get()"""
        self.net.set({'tau_m':43.21})
        assert HocToPy.get('%s.object(7).tau_m' % self.net.label, 'float') == 43.21
    
    def testSetFromPair(self):
        """Population.set(): A parameter set as a string,value pair should be retrievable using HocToPy.get()"""
        self.net.set('tau_m',12.34)
        assert HocToPy.get('%s.object(6).tau_m' % self.net.label, 'float') == 12.34
    
    def testSetInvalidFromPair(self):
        """Population.set(): Trying to set an invalid value for a parameter should raise an exception."""
        self.assertRaises(common.InvalidParameterValueError, self.net.set, 'tau_m', [])
    
    def testSetInvalidFromDict(self):
        """Population.set(): When any of the parameters in a dict have invalid values, then an exception should be raised.
           There is no guarantee that the valid parameters will be set."""
        self.assertRaises(common.InvalidParameterValueError, self.net.set, {'v_thresh':'hello','tau_m':56.78})
    
    def testSetNonexistentFromPair(self):
        """Population.set(): Trying to set a nonexistent parameter should raise an exception."""
        self.assertRaises(common.NonExistentParameterError, self.net.set, 'tau_foo', 10.0)
    
    def testSetNonexistentFromDict(self):
        """Population.set(): When some of the parameters in a dict are inexistent, an exception should be raised.
           There is no guarantee that the existing parameters will be set."""
        self.assertRaises(common.NonExistentParameterError, self.net.set, {'tau_foo': 10.0, 'tau_m': 21.0})
    
    def testSetWithNonStandardModel(self):
        """Population.set(): Parameters set in a dict should all be retrievable using HocToPy.get()"""
        self.net2.set({'tau_m':43.21})
        assert HocToPy.get('%s.object(2).tau_m' % self.net2.label, 'float') == 43.21
        
    def testTSet(self):
        """Population.tset(): The valueArray passed should be retrievable using HocToPy.get() on all nodes."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.net.tset('i_offset', array_in)
        array_out = numpy.zeros((3,3),float)
        for i in 0,1,2:
            for j in 0,1,2:
                array_out[i,j]= HocToPy.get('%s.object(%d).stim.amp' % (self.net.label,3*i+j),'float')
        assert numpy.equal(array_in, array_out).all()
    
    def testTSetInvalidDimensions(self):
        """Population.tset(): If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.assertRaises(common.InvalidDimensionsError, self.net.tset, 'i_offset', array_in)
    
    def testTSetInvalidValues(self):
        """Population.tset(): If some of the values in the valueArray are invalid, should raise an exception."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,'apples']])
        self.assertRaises(common.InvalidParameterValueError, self.net.tset, 'i_offset', array_in)
        
    def testRSetNative(self):
        """Population.rset(): with native rng. This is difficult to test, so for now just require that all values retrieved should be different. Later, could calculate distribution and assert that the difference between sample and theoretical distribution is less than some threshold."""
        self.net.rset('tau_m',
                      random.RandomDistribution(rng=random.NativeRNG(),
                                                distribution='uniform',
                                                parameters=(10.0,30.0)))
        self.assertNotEqual(HocToPy.get('%s.object(5).tau_m' % self.net.label),
                            HocToPy.get('%s.object(6).tau_m' % self.net.label))
        
    def testRSetNumpy(self):
        """Population.rset(): with numpy rng."""
        rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.net.rset('cm',rd1)
        output_values = numpy.zeros((3,3),numpy.float)
        for i in 0,1,2:
            for j in 0,1,2:
                output_values[i,j] = HocToPy.get('%s.object(%d).cell.cm' % (self.net.label,3*i+j),'float')
        input_values = rd2.next(9)
        output_values = output_values.reshape((9,))
        for i in range(9):
            self.assertAlmostEqual(input_values[i],output_values[i],places=5)
        
    def testRSetNative(self):
        """Population.rset(): with native rng."""
        rd1 = random.RandomDistribution(rng=random.NativeRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NativeRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.net.rset('cm',rd1)
        output_values_1 = numpy.zeros((3,3),numpy.float)
        output_values_2 = numpy.zeros((3,3),numpy.float)
        for i in 0,1,2:
            for j in 0,1,2:
                output_values_1[i,j] = HocToPy.get('%s.object(%d).cell.cm' % (self.net.label,3*i+j),'float')
                
        self.net.rset('cm',rd2)
        for i in 0,1,2:
            for j in 0,1,2:
                output_values_2[i,j] = HocToPy.get('%s.object(%d).cell.cm' % (self.net.label,3*i+j),'float')

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
	self.pop1 = neuron.Population((3,3), neuron.SpikeSourcePoisson,{'rate': 20})
	self.pop2 = neuron.Population((3,3), neuron.IF_curr_alpha)

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
        neuron.running = False
	neuron.run(simtime)
	self.pop1.printSpikes("temp_neuron.ras")
	rate = self.pop1.meanSpikeCount()*1000/simtime
	assert (20*0.8 < rate) and (rate < 20*1.2)

    def testPotentialRecording(self):
	"""Population.record_v() and Population.print_v(): not a full test, just checking 
	# there are no Exceptions raised."""
	rng = NumpyRNG(123)
	v_reset  = -65.0
	v_thresh = -50.0
	uniformDistr = RandomDistribution(rng,'uniform',[v_reset,v_thresh])
	self.pop2.randomInit(uniformDistr)
	self.pop2.record_v([self.pop2[0,0], self.pop2[1,1]])
	simtime = 10.0
        neuron.running = False
        neuron.run(simtime)
	self.pop2.print_v("temp_neuron.v")

# ==============================================================================
class PopulationOtherTest(unittest.TestCase): # to write later
    """Tests of the randomInit() method of the Population class."""
    pass

# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Projection class."""
        
    def setUp(self):
        neuron.Population.nPop = 0
        neuron.Projection.nProj = 0
        self.target33    = neuron.Population((3,3),neuron.IF_curr_alpha)
        self.target6     = neuron.Population((6,),neuron.IF_curr_alpha)
        self.source5     = neuron.Population((5,),neuron.SpikeSourcePoisson)
        self.source22    = neuron.Population((2,2),neuron.SpikeSourcePoisson)
        self.source33    = neuron.Population((3,3),neuron.SpikeSourcePoisson)
        self.expoisson33 = neuron.Population((3,3),neuron.SpikeSourcePoisson,{'rate': 100})
        
    def testAllToAll(self):
        """For all connections created with "allToAll" it should be possible to obtain the weight using pyneuron.getWeight()"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, 'allToAll')
                prj1.setWeights(1.234)
                weights = []
                for connection_id in prj1.connections:
                    weights.append(HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id)), 'float'))
                assert weights == [1.234]*len(prj1)
        
    def testFixedProbability(self):
        """For all connections created with "fixedProbability" it should be possible to obtain the weight using pyneuron.getWeight()"""
        distrib_Numpy = RandomDistribution(NumpyRNG(12345),'uniform',(0,1)) 
        distrib_Native= RandomDistribution(NativeRNG(12345),'uniform',(0,1)) 
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, 'fixedProbability', 0.5, distrib_Numpy)
                prj2 = neuron.Projection(srcP, tgtP, 'fixedProbability', 0.5, distrib_Native)
                assert (0 < len(prj1) < len(srcP)*len(tgtP)) and (0 < len(prj2) < len(srcP)*len(tgtP))
                
    def testoneToOne(self):
        """For all connections created with "OneToOne" it should be possible to obtain the weight using pyneuron.getWeight()"""
        prj1 = neuron.Projection(self.source33, self.target33, 'oneToOne')
        assert len(prj1.connections) == self.source33.size
     
    def testdistantDependentProbability(self):
        """For all connections created with "distanceDependentProbability" it should be possible to obtain the weight using pyneuron.getWeight()"""
        # Test should be improved..."
        distrib_Numpy = RandomDistribution(NumpyRNG(12345),'uniform',(0,1)) 
        distrib_Native= RandomDistribution(NativeRNG(12345),'uniform',(0,1)) 
        prj1 = neuron.Projection(self.source33, self.target33, 'distanceDependentProbability',{'d_expression' : 'exp(-d)'}, distrib_Numpy)
        prj2 = neuron.Projection(self.source33, self.target33, 'distanceDependentProbability',{'d_expression' : 'd < 0.5'}, distrib_Native)
        assert (0 < len(prj1) < len(self.source33)*len(self.target33)) and (0 < len(prj2) < len(self.source33)*len(self.target33))
        
    def testSaveAndLoad(self):
        prj1 = neuron.Projection(self.source33, self.target33, 'oneToOne')
        prj1.setDelays(1)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp")
        prj2 = neuron.Projection(self.source33, self.target33, 'fromFile',"connections.tmp")
        w1 = []; w2 = []; d1 = []; d2 = [];
        # For a connections scheme saved and reloaded, we test if the connections, their weights and their delays
        # are equal.
        for connection_id in prj1.connections:
            w1.append(HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id))))
            w2.append(HocToPy.get('%s.object(%d).weight' % (prj2.label,prj2.connections.index(connection_id))))
            d1.append(HocToPy.get('%s.object(%d).delay' % (prj1.label,prj1.connections.index(connection_id))))
            d2.append(HocToPy.get('%s.object(%d).delay' % (prj2.label,prj2.connections.index(connection_id))))
        assert (w1 == w2) and (d1 == d2)
          


class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), setThreshold(),
#       randomizeWeights() and randomizeDelays() methods of the Projection class."""

    def setUp(self):
        self.target   = neuron.Population((3,3),neuron.IF_curr_alpha)
        self.target   = neuron.Population((3,3),neuron.IF_curr_alpha)
        self.source   = neuron.Population((3,3),neuron.SpikeSourcePoisson,{'rate': 100})
        self.distrib_Numpy = RandomDistribution(NumpyRNG(12345),'uniform',(0,1)) 
        self.distrib_Native= RandomDistribution(NativeRNG(12345),'uniform',(0,1)) 
        
    def testsetWeights(self):
        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
        prj1.setWeights(2.345)
        weights = []
        for connection_id in prj1.connections:
            weights.append(HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id))))
        result = 2.345*numpy.ones(len(prj1.connections))
        assert (weights == result.tolist())
        
    def testsetDelays(self):
        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
        prj1.setDelays(2.345)
        delays = []
        for connection_id in prj1.connections:
            delays.append(HocToPy.get('%s.object(%d).delay' % (prj1.label,prj1.connections.index(connection_id))))
        result = 2.345*numpy.ones(len(prj1.connections))
        assert (delays == result.tolist())
        
    def testrandomizeWeights(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
        prj2 = neuron.Projection(self.source, self.target, 'allToAll')
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        w1 = []; w2 = []; w3 = []; w4 = []
        for connection_id in prj1.connections:
            w1.append(HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id))))
            w2.append(HocToPy.get('%s.object(%d).weight' % (prj2.label,prj1.connections.index(connection_id))))
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        for connection_id in prj1.connections:
            w3.append(HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id))))
            w4.append(HocToPy.get('%s.object(%d).weight' % (prj2.label,prj1.connections.index(connection_id))))  
        self.assertNotEqual(w1,w3) and self.assertNotEqual(w2,w4) 
        
    def testrandomizeDelays(self):
        # The probability of having two consecutive delays vector that are equal should be 0
        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
        prj2 = neuron.Projection(self.source, self.target, 'allToAll')
        prj1.randomizeDelays(self.distrib_Numpy)
        prj2.randomizeDelays(self.distrib_Native)
        d1 = []; d2 = []; d3 = []; d4 = []
        for connection_id in prj1.connections:
            d1.append(HocToPy.get('%s.object(%d).delay' % (prj1.label,prj1.connections.index(connection_id))))
            d2.append(HocToPy.get('%s.object(%d).delay' % (prj2.label,prj1.connections.index(connection_id))))
        prj1.randomizeDelays(self.distrib_Numpy)
        prj2.randomizeDelays(self.distrib_Native)
        for connection_id in prj1.connections:
            d3.append(HocToPy.get('%s.object(%d).delay' % (prj1.label,prj1.connections.index(connection_id))))
            d4.append(HocToPy.get('%s.object(%d).delay' % (prj2.label,prj1.connections.index(connection_id))))  
        self.assertNotEqual(d1,d3) and self.assertNotEqual(d2,d4) 
               
        
    # If STDP works, a strong stimulation with only LTP should increase the mean weight
    def testSetupSTDP(self):
        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
        prj1.setDelays(2)
        prj1.setWeights(0.5)
        STDP_params = {'aLTP'       : 1,
                       'aLTD'       : 0}
        prj1.setupSTDP("StdwaSA", STDP_params)
        mean_weight_before = 0
        for connection_id in prj1.connections:
            mean_weight_before += HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id)), 'float')        
        mean_weight_before = float(mean_weight_before/len(prj1.connections))  
        simtime = 100
        neuron.running = False
        run(simtime)
        mean_weight_after = 0
        for connection_id in prj1.connections:
            mean_weight_after += HocToPy.get('%s.object(%d).weight' % (prj1.label,prj1.connections.index(connection_id)), 'float')     
        mean_weight_after = float(mean_weight_after/len(prj1.connections))
        assert (mean_weight_before < mean_weight_after)
        
    def testSetAndGetID(self):
        # Small test to see if the ID class is working
        self.target[0,2].set({'tau_m' : 15.1})
        assert (self.target[0,2].get('tau_m') == 15.1)
        
    def testSetAndGetPositionID(self):
        # Small test to see if the position of the ID class is working
        self.target[0,2].setPosition((0.5,1.5))
        assert (self.target[0,2].getPosition() == (0.5,1.5))
        
    def testSetTopographicDelay(self):
        # We fix arbitrarily the positions of 2 cells in 2 populations and check 
        # the topographical delay between them is linked to the distance
        self.source[0,0].setPosition((0,0))
        self.target[2,2].setPosition((0,10))
        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
        rule="5.432*d"
        prj1.setTopographicDelays(rule)
        for connection_id in range(len(prj1.connections)):
            src = prj1.connections[connection_id][0]
            tgt = prj1.connections[connection_id][1]
            if (src == self.source[0,0]) and (tgt == self.target[2,2]):
                delay = HocToPy.get('%s.object(%d).delay' % (prj1.label,prj1.connections.index(prj1.connections[connection_id])), 'float')
        assert (delay == 54.32)

#class ProjectionConnectionTest(unittest.TestCase):
#    """Tests of the connection attribute and connections() method of the Projection class."""
#    
#    def setUp(self):
#        neuron.Population.nPop = 0
#        self.pop1 = neuron.Population((5,),neuron.IF_curr_alpha)
#        self.pop2 = neuron.Population((4,4),neuron.IF_curr_alpha)    
#        self.pop3 = neuron.Population((3,3,3),neuron.IF_curr_alpha)
#        self.prj23 = neuron.Projection(self.pop2,self.pop3,"allToAll")
#        self.prj11 = neuron.Projection(self.pop1,self.pop1,"fixedProbability",0.5)
#        
#    def testFullAddress(self):
#        assert self.prj23.connection[(3,1),(2,0,1)] == "[3][1][2][0][1]"
#        assert self.prj23.connection[(3,3),(2,2,2)] == "[3][3][2][2][2]"
#        
#    def testPreIDPostID(self):
#        assert self.prj23.connection[0,0] == "[0][0][0][0][0]"
#        assert self.prj23.connection[0,26] == "[0][0][2][2][2]"
#        assert self.prj23.connection[0,25] == "[0][0][2][2][1]"
#        assert self.prj23.connection[15,0] == "[3][3][0][0][0]"
#        assert self.prj23.connection[14,0] == "[3][2][0][0][0]"
#        assert self.prj23.connection[13,19] == "[3][1][2][0][1]"
#        
#    def testSingleID(self):
#        assert self.prj23.connection[0] == "[0][0][0][0][0]"
#        assert self.prj23.connection[26] == "[0][0][2][2][2]"
#        assert self.prj23.connection[25] == "[0][0][2][2][1]"
#        assert self.prj23.connection[27] == "[0][1][0][0][0]"
#        assert self.prj23.connection[53] == "[0][1][2][2][2]"
#        assert self.prj23.connection[52] == "[0][1][2][2][1]"
#        assert self.prj23.connection[431] == "[3][3][2][2][2]"
#        assert self.prj23.connection[377] == "[3][1][2][2][2]"
#        assert self.prj23.connection[370] == "[3][1][2][0][1]"
#        
#        assert self.prj11.connection[0] == "[0][0]"

class IDTest(unittest.TestCase):
    """Tests of the ID class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.pop = neuron.Population((5,),neuron.IF_curr_alpha,{'tau_m':10.0})
    
    def testIDSet(self):
        self.pop[3].set('tau_m',20.0)
        self.assertEqual(HocToPy.get('%s.object(3).tau_m' % self.pop.label, 'float'), 20.0)
        self.assertEqual(HocToPy.get('%s.object(1).tau_m' % self.pop.label, 'float'), 10.0)

if __name__ == "__main__":
    sys.argv = ['./nrnpython']
    neuron.setup()
    unittest.main()
    neuron.end()
    