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
        ifcells = neuron.create(neuron.IF_curr_alpha, n=10)
        assert ifcells == range(0,10), 'Failed to create 10 standard cells'
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        neuron.hoc_comment('=== CreationTest.testCreateStandardCellsWithNegative_n ===')
        self.assertRaises(AssertionError, neuron.create, neuron.IF_curr_alpha, n=-1)
       
    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with the top-level HocObject"""
        neuron.hoc_comment('=== CreationTest.testCreateStandardCellWithParams ===')
        ifcell = neuron.create(neuron.IF_curr_alpha,{'tau_syn_E':3.141592654})
        #self.assertAlmostEqual(HocToPy.get('cell%d.esyn.tau' % ifcell, 'float'), 3.141592654, places=5)
        try:
            self.assertAlmostEqual(getattr(h, 'cell%d' % ifcell).esyn.tau, 3.141592654, places=5)
        except AttributeError: # if the cell is not on that node
            pass
        
    
    def testCreateNEURONCell(self):
        """create(): First cell created should have index 0."""
        neuron.hoc_comment('=== CreationTest.testCreateNEURONCell ===')
        ifcell = neuron.create('StandardIF',{'syn_type':'current','syn_shape':'exp'})
        assert ifcell == 0, 'Failed to create NEURON-specific cell'
    
    def testCreateInvalidCell(self):
        """create(): Trying to create a cell type which is not a standard cell or
        valid native cell should raise a HocError."""
        self.assertRaises(HocError, neuron.create, 'qwerty', n=10)
    
    def testCreateWithInvalidParameter(self):
        """create(): Creating a cell with an invalid parameter should raise an Exception."""
        self.assertRaises(common.NonExistentParameterError, neuron.create, neuron.IF_curr_alpha, {'tau_foo':3.141592654})        


# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        neuron.hoc_comment("=== ConnectionTest.setUp() ===")
        self.postcells = neuron.create(neuron.IF_curr_alpha, n=3)
        self.precells = neuron.create(neuron.SpikeSourcePoisson, n=5)
        
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
        conn = neuron.connect(self.precells[0], self.postcells[0])
        # conn will be an empty list if it does not exist on that node
        assert conn == [0] or conn == [], 'Error creating connection, conn=%s' % conn
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        neuron.hoc_comment("=== ConnectionTest.testConnectTwoCellsWithWeight() ===")
        conn_id = neuron.connect(self.precells[0], self.postcells[0], weight=0.1234)
        if conn_id:
            weight = h.netconlist.object(conn_id[0]).weight[0]
            assert weight == 0.1234, "Weight set (0.1234) does not match weight retrieved (%s)" % weight
    
    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        conn_id = neuron.connect(self.precells[0], self.postcells[0], delay=4.321)
        if conn_id:
            delay = h.netconlist.object(conn_id[0]).delay
            assert delay == 4.321, "Delay set (4.321) does not match delay retrieved (%s)." % delay
    
    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n,
        each element being the id number of a netcon."""
        connlist = neuron.connect(self.precells, self.postcells[0])
        # connections are only created on the node containing the post-syn
        assert connlist == range(0, len(self.precells)) or connlist == [], connlist
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        connlist = neuron.connect(self.precells[0], self.postcells)
        cells_on_this_node = len([i for i in self.postcells if i in neuron.gidlist])
        assert connlist == range(cells_on_this_node)
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        connlist = neuron.connect(self.precells, self.postcells)
        cells_on_this_node = len([i for i in self.postcells if i in neuron.gidlist])
        expected_connlist = range(cells_on_this_node*len(self.precells))
        self.assert_(connlist == expected_connlist, "%s != %s" % (connlist, expected_connlist))
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        connlist = neuron.connect(self.precells, self.postcells, p=0.5)
        cells_on_this_node = len([i for i in self.postcells if i in neuron.gidlist])
        assert 0 < len(connlist) < len(self.precells)*cells_on_this_node, 'Number of connections is %d: this is very unlikely (although possible).' % len(connlist)
    
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
        self.cells = neuron.create(neuron.IF_curr_exp, n=10)
        
    def testSetFloat(self):
        neuron.hoc_comment("=== SetValueTest.testSetFloat() ===")
        neuron.set(self.cells, 'tau_m',35.7)
        for cell in self.cells:
            try:
                assert getattr(h, 'cell%d' % cell).tau_m == 35.7
            except AttributeError: # if cell is not on this node
                pass
  
    #def testSetString(self):
    #    neuron.set(self.cells, neuron.IF_curr_exp,'param_name','string_value')
    ## note we don't currently have any models with string parameters, so
    ## this is all commented out
    #    for cell in self.cells:
    #        assert HocToPy.get('cell%d.param_name' % cell, 'string') == 'string_value'

    def testSetDict(self):
        neuron.set(self.cells, {'tau_m':35.7, 'tau_syn_E':5.432})
        for cell in self.cells:
            try:
                hoc_cell = getattr(h, 'cell%d' % cell)
                assert hoc_cell.tau_e == 5.432
                assert hoc_cell.tau_m == 35.7
            except AttributeError: # if cell is not on this node
                pass
            
    def testSetNonExistentParameter(self):
        # note that although syn_shape is added to the parameter dict when creating
        # an IF_curr_exp, it is not a valid parameter to be changed later.
        self.assertRaises(common.NonExistentParameterError, neuron.set, self.cells, 'syn_shape', 'alpha')

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
        net = neuron.Population((3,3), neuron.IF_curr_alpha)
        n_cells = getattr(h, net.label).count()
        n_cells_lower = int(getattr(h, net.label).count())
        # round-robin distribution
        assert 9/neuron.nhost <= n_cells_lower <= 9/neuron.nhost+1, "%d not between %d and %d" % (n_cells_lower, 9/neuron.nhost, 9/neuron.nhost+1)
    
    def testInitWithParams(self):
        """Population.__init__(): Parameters set on creation should be the same as
        retrieved with the top-level HocObject"""
        net = neuron.Population((3,3), neuron.IF_curr_alpha,{'tau_syn_E':3.141592654})
        cell_list = getattr(h, net.label)
        for i in range(int(cell_list.count())):
            tau_syn = cell_list.object(i).esyn.tau
            self.assertAlmostEqual(tau_syn, 3.141592654, places=5)
    
    def testInitWithLabel(self):
        """Population.__init__(): A label set on initialisation should be retrievable with the Population.label attribute."""
        net = neuron.Population((3,3), neuron.IF_curr_alpha, label='iurghiushrg')
        assert net.label == 'iurghiushrg'
    
    def testInvalidCellType(self):
        """Population.__init__(): Trying to create a cell type which is not a StandardCell
        or a valid neuron model should raise a HocError."""
        self.assertRaises(neuron.HocError, neuron.Population, (3,3), 'qwerty', {})
        
    def testInitWithNonStandardModel(self):
        """Population.__init__(): the cell list in hoc should have the same length as the population size."""
        net = neuron.Population((3,3), 'StandardIF', {'syn_type':'current', 'syn_shape':'exp'})
        n_cells = getattr(h, net.label).count()
        n_cells_lower = int(getattr(h, net.label).count())
        # round-robin distribution
        assert 9/neuron.nhost <= n_cells_lower <= 9/neuron.nhost+1, "%d not between %d and %d" % (n_cells_lower, 9/neuron.nhost, 9/neuron.nhost+1)
    

# ==============================================================================
class PopulationIndexTest(unittest.TestCase):
    """Tests of the Population class indexing."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.net1 = neuron.Population((10,), neuron.IF_curr_alpha)
        self.net2 = neuron.Population((2,4,3), neuron.IF_curr_exp)
        self.net3 = neuron.Population((2,2,1), neuron.SpikeSourceArray)
        self.net4 = neuron.Population((1,2,1), neuron.SpikeSourceArray)
        self.net5 = neuron.Population((3,3), neuron.IF_cond_alpha)
    
    def testValidIndices(self):
        for i in range(10):
            self.assertEqual((i,), self.net1.locate(self.net1[i]))

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
        self.net1 = neuron.Population((10,), neuron.IF_curr_alpha)
        self.net2 = neuron.Population((2,4,3), neuron.IF_curr_exp)
        self.net3 = neuron.Population((2,2,1), neuron.SpikeSourceArray)
        self.net4 = neuron.Population((1,2,1), neuron.SpikeSourceArray)
        self.net5 = neuron.Population((3,3), neuron.IF_cond_alpha)
        
    def testIter(self):
        """This needs more thought for the distributed case."""
        for net in self.net1, self.net2:
            ids = [i for i in net]
            self.assertEqual(ids, net.gidlist)
            self.assert_(isinstance(ids[0], neuron.ID))
            
    def testAddressIter(self):
        for net in self.net1, self.net2:
            for id, addr in zip(net.ids(), net.addresses()):
                self.assertEqual(id, net[addr])
                self.assertEqual(addr, net.locate(id))
            
    
# ==============================================================================
class PopulationSetTest(unittest.TestCase):
        
    def setUp(self):
        neuron.Population.nPop = 0
        self.net = neuron.Population((3,3), neuron.IF_curr_alpha)
        self.net2 = neuron.Population((5,),'StandardIF',{'syn_type':'current','syn_shape':'exp'})
    
    def testSetFromDict(self):
        """Population.set(): Parameters set in a dict should all be retrievable using the top-level HocObject"""
        self.net.set({'tau_m':43.21})
        cell_list = getattr(h, self.net.label)
        for i in range(int(cell_list.count())):
            assert cell_list.object(i).tau_m == 43.21
    
    def testSetFromPair(self):
        """Population.set(): A parameter set as a string, value pair should be retrievable using the top-level HocObject"""
        self.net.set('tau_m',12.34)
        cell_list = getattr(h, self.net.label)
        for i in range(int(cell_list.count())):
            assert cell_list.object(i).tau_m == 12.34
    
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
        """Population.set(): Parameters set in a dict should all be retrievable using the top-level HocObject"""
        self.net2.set({'tau_m':43.21})
        cell_list = getattr(h, self.net2.label)
        for i in range(int(cell_list.count())):
            assert cell_list.object(i).tau_m == 43.21
        
    def testTSet(self):
        """Population.tset(): The valueArray passed should be retrievable using the top-level HocObject on all nodes."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.net.tset('i_offset', array_in)
        array_out = numpy.zeros((3,3), float)
        hoc_net = getattr(h, self.net.label)
        for i in 0,1,2:
            for j in 0,1,2:
                id = 3*i+j
                if id in self.net.gidlist:
                    list_index = self.net.gidlist.index(id)
                    cell = hoc_net.object(list_index)
                    array_out[i, j] = cell.stim.amp
                else:
                    array_out[i, j] = array_in[i, j]
        assert numpy.equal(array_in, array_out).all(), array_out
    
    def testTSetInvalidDimensions(self):
        """Population.tset(): If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.assertRaises(common.InvalidDimensionsError, self.net.tset, 'i_offset', array_in)
    
    def testTSetInvalidValues(self):
        """Population.tset(): If some of the values in the valueArray are invalid, should raise an exception."""
        array_in = numpy.array([['potatoes','carrots','peas'],['dogs','cats','mice'],['oranges','bananas','apples']])
        self.assertRaises(common.InvalidParameterValueError, self.net.tset, 'i_offset', array_in)
        
    def testRSetNative1(self):
        """Population.rset(): with native rng. This is difficult to test, so for
        now just require that all values retrieved should be different.
        Later, could calculate distribution and assert that the difference between
        sample and theoretical distribution is less than some threshold."""
        self.net.rset('tau_m',
                      random.RandomDistribution(rng=random.NativeRNG(),
                                                distribution='uniform',
                                                parameters=(10.0,30.0)))
        hoc_net = getattr(h, self.net.label)
        self.assertNotEqual(hoc_net.object(0).tau_m,
                            hoc_net.object(1).tau_m)
        
    def testRSetNumpy(self):
        """Population.rset(): with numpy rng."""
        rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.net.rset('cm', rd1)
        output_values = numpy.zeros((3,3), numpy.float)
        hoc_net = getattr(h, self.net.label)
        for i in 0,1,2:
            for j in 0,1,2:
                id = 3*i+j
                if id in self.net.gidlist:
                    list_index = self.net.gidlist.index(id)
                    output_values[i, j] = hoc_net.object(list_index).cell(0.5).cm
        input_values = rd2.next(9)
        output_values = output_values.reshape((9,))
        for i in range(9):
            if i in self.net.gidlist:
                self.assertAlmostEqual(input_values[i], output_values[i], places=5)
        
    def testRSetNative2(self):
        """Population.rset(): with native rng."""
        rd1 = random.RandomDistribution(rng=random.NativeRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NativeRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.net.rset('cm', rd1)
        output_values_1 = numpy.zeros((3,3), numpy.float)
        output_values_2 = numpy.zeros((3,3), numpy.float)
        hoc_net = getattr(h, self.net.label)
        print hoc_net.count()
        for i in 0,1,2:
            for j in 0,1,2:
                id = 3*i+j
                if id in self.net.gidlist:
                    list_index = self.net.gidlist.index(id)
                    output_values_1[i, j] = hoc_net.object(list_index).cell(0.5).cm
        self.net.rset('cm', rd2)
        for i in 0,1,2:
            for j in 0,1,2:
                id = 3*i+j
                if id in self.net.gidlist:
                    list_index = self.net.gidlist.index(id)
                    output_values_2[i, j] = hoc_net.object(list_index).cell(0.5).cm
        output_values_1 = output_values_1.reshape((9,))
        output_values_2 = output_values_2.reshape((9,))
        for i in range(9):
            self.assertAlmostEqual(output_values_1[i], output_values_2[i], places=5)    

# ==============================================================================
class PopulationRecordTest(unittest.TestCase): # to write later
    """Tests of the record(), record_v(), printSpikes(), print_v() and
       meanSpikeCount() methods of the Population class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
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
        """Population.record(n, rng): not a full test, just checking there are no Exceptions raised."""
	self.pop1.record(5, random.NumpyRNG())
        
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
	self.pop1.printSpikes("temp_neuron.ras", gather=True)
        rate = self.pop1.meanSpikeCount()*1000/simtime
        if neuron.myid == 0: # only on master node
            assert (20*0.8 < rate) and (rate < 20*1.2), "rate is %s" % rate

    def testPotentialRecording(self):
	"""Population.record_v() and Population.print_v(): not a full test, just checking 
	# there are no Exceptions raised."""
	rng = NumpyRNG(123)
	v_reset  = -65.0
	v_thresh = -50.0
	uniformDistr = RandomDistribution(rng=rng, distribution='uniform', parameters=[v_reset, v_thresh])
	self.pop2.randomInit(uniformDistr)
	self.pop2.record_v([self.pop2[0,0], self.pop2[1,1]])
	simtime = 10.0
        neuron.running = False
        neuron.run(simtime)
	self.pop2.print_v("temp_neuron.v", gather=True)

    def testRecordWithSpikeTimesGreaterThanSimTime(self):
        """
        If a `SpikeSourceArray` is initialized with spike times greater than the
        simulation time, only those spikes that actually occurred should be
        written to file or returned by getSpikes().
        """
        spike_times = numpy.arange(10.0, 200.0, 10.0)
        spike_source = neuron.Population(1, neuron.SpikeSourceArray, {'spike_times': spike_times})
        spike_source.record()
        neuron.running = False
        neuron.run(100.0)
        spikes = spike_source.getSpikes()[:,1]
        if neuron.myid == 0:
            self.assert_( max(spikes) == 100.0, str(spikes) )

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
        self.target33    = neuron.Population((3,3), neuron.IF_curr_alpha)
        self.target6     = neuron.Population((6,), neuron.IF_curr_alpha)
        self.source5     = neuron.Population((5,), neuron.SpikeSourcePoisson)
        self.source22    = neuron.Population((2,2), neuron.SpikeSourcePoisson)
        self.source33    = neuron.Population((3,3), neuron.SpikeSourcePoisson)
        self.expoisson33 = neuron.Population((3,3), neuron.SpikeSourcePoisson,{'rate': 100})
        
    def testAllToAll(self):
        """For all connections created with "allToAll" it should be possible to
        obtain the weight using the top-level HocObject"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, neuron.AllToAllConnector())
                prj2 = neuron.Projection(srcP, tgtP, neuron.AllToAllConnector())
                prj1.setWeights(1.234)
                prj2.setWeights(1.234)
                for prj in prj1, prj2:
                    hoc_list = getattr(h, prj.hoc_label)
                    weights = []
                    for connection_id in prj.connections:
                        weights.append(hoc_list.object(prj.connections.index(connection_id)).weight[0])
                    assert weights == [1.234]*len(prj)
            
    def testFixedProbability(self):
        """For all connections created with "fixedProbability"..."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj2 = neuron.Projection(srcP, tgtP, FixedProbabilityConnector(0.5), rng=NativeRNG(12345))
                prj3 = neuron.Projection(srcP, tgtP, FixedProbabilityConnector(0.5), rng=NumpyRNG(12345))
                assert (0 < len(prj2) < len(srcP)*len(tgtP)) \
                       and (0 < len(prj3) < len(srcP)*len(tgtP))
                
    def testOneToOne(self):
        """For all connections created with "OneToOne" ..."""
        prj = neuron.Projection(self.source33, self.target33, neuron.OneToOneConnector())
        assert len(prj.connections) == len(self.target33.gidlist), prj.connections
     
    def testDistanceDependentProbability(self):
        """For all connections created with "distanceDependentProbability"..."""
        # Test should be improved..."
        for rngclass in (NumpyRNG, NativeRNG):
            for expr in ('exp(-d)', 'd < 0.5'):
                prj = neuron.Projection(self.source33, self.target33,
                                         neuron.DistanceDependentProbabilityConnector(d_expression=expr),
                                         rng=rngclass(12345))
                assert (0 < len(prj) < len(self.source33)*len(self.target33)) 

    def testFixedNumberPre(self):
        c1 = neuron.FixedNumberPreConnector(10)
        c2 = neuron.FixedNumberPreConnector(3)
        c3 = neuron.FixedNumberPreConnector(random.RandomDistribution('poisson',[5]))
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                for c in c1, c2:
                    prj1 = neuron.Projection(srcP, tgtP, c)
                    self.assertEqual(len(prj1.connections), c.n*len(tgtP))
                prj3 = neuron.Projection(srcP, tgtP, c3) # just a test that no Exceptions are raised

    def testFixedNumberPost(self):
        c1 = neuron.FixedNumberPostConnector(10)
        c2 = neuron.FixedNumberPostConnector(3)
        c3 = neuron.FixedNumberPostConnector(random.RandomDistribution('poisson',[5]))
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                for c in c1, c2:
                    prj1 = neuron.Projection(srcP, tgtP, c)
                    self.assertEqual(len(prj1.connections), c.n*len(srcP))
                prj2 = neuron.Projection(srcP, tgtP, c3) # just a test that no Exceptions are raised

    def testSaveAndLoad(self):
        prj1 = neuron.Projection(self.source33, self.target33, neuron.OneToOneConnector())
        prj1.setDelays(1)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp", gather=False)
        #prj2 = neuron.Projection(self.source33, self.target33, 'fromFile',"connections.tmp")
        if neuron.num_processes() > 1:
            distributed = True
        else:
            distributed = False
        prj3 = neuron.Projection(self.source33, self.target33, neuron.FromFileConnector("connections.tmp",
                                                                                        distributed=distributed))
        w1 = []; w2 = []; w3 = []; d1 = []; d2 = []; d3 = []
        # For a connections scheme saved and reloaded, we test if the connections, their weights and their delays
        # are equal.
        hoc_list1 = getattr(h, prj1.hoc_label)
        #hoc_list2 = getattr(h, prj2.hoc_label)
        hoc_list3 = getattr(h, prj3.hoc_label)
        for connection_id in prj1.connections:
            w1.append(hoc_list1.object(prj1.connections.index(connection_id)).weight[0])
            #w2.append(hoc_list2.object(prj2.connections.index(connection_id)).weight[0])
            w3.append(hoc_list3.object(prj3.connections.index(connection_id)).weight[0])
            d1.append(hoc_list1.object(prj1.connections.index(connection_id)).delay)
            #d2.append(hoc_list2.object(prj2.connections.index(connection_id)).delay)
            d3.append(hoc_list3.object(prj3.connections.index(connection_id)).delay)
        #assert (w1 == w2 == w3) and (d1 == d2 == d3)
        assert (w1 == w3) and (d1 == d3)
          
    def testSettingDelays(self):
        """Delays should be set correctly when using a Connector object."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, neuron.AllToAllConnector(delays=0.321))
                hoc_list = getattr(neuron.h, prj1.hoc_label)
                assert hoc_list.object(0).delay == 0.321, "Delay should be 0.321, actually %g" % hoc_list.object(0).delay

class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), randomizeWeights() and
    randomizeDelays() methods of the Projection class."""

    def setUp(self):
        self.target   = neuron.Population((3,3), neuron.IF_curr_alpha)
        self.source   = neuron.Population((3,3), neuron.SpikeSourcePoisson,{'rate': 200})
        self.distrib_Numpy = RandomDistribution(rng=NumpyRNG(12345), distribution='uniform', parameters=(0,1)) 
        self.distrib_Native= RandomDistribution(rng=NativeRNG(12345), distribution='uniform', parameters=(0,1)) 
        
    def testSetWeights(self):
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.setWeights(2.345)
        weights = []
        hoc_list = getattr(h, prj1.hoc_label)
        for connection_id in prj1.connections:
            #weights.append(HocToPy.get('%s.object(%d).weight' % (prj1.hoc_label, prj1.connections.index(connection_id))))
            weights.append(hoc_list.object(prj1.connections.index(connection_id)).weight[0])
        result = 2.345*numpy.ones(len(prj1.connections))
        assert (weights == result.tolist())
        
    def testSetDelays(self):
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.setDelays(2.345)
        delays = []
        hoc_list = getattr(h, prj1.hoc_label)
        for connection_id in prj1.connections:
            #delays.append(HocToPy.get('%s.object(%d).delay' % (prj1.hoc_label, prj1.connections.index(connection_id))))
            delays.append(hoc_list.object(prj1.connections.index(connection_id)).delay)
        result = 2.345*numpy.ones(len(prj1.connections))
        assert (delays == result.tolist())
        
    def testRandomizeWeights(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj2 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        w1 = []; w2 = []; w3 = []; w4 = []
        hoc_list1 = getattr(h, prj1.hoc_label)
        hoc_list2 = getattr(h, prj2.hoc_label)
        for connection_id in prj1.connections:
            #w1.append(HocToPy.get('%s.object(%d).weight' % (prj1.hoc_label, prj1.connections.index(connection_id))))
            #w2.append(HocToPy.get('%s.object(%d).weight' % (prj2.hoc_label, prj1.connections.index(connection_id))))
            w1.append(hoc_list1.object(prj1.connections.index(connection_id)).weight[0])
            w2.append(hoc_list2.object(prj2.connections.index(connection_id)).weight[0])
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        for connection_id in prj1.connections:
            #w3.append(HocToPy.get('%s.object(%d).weight' % (prj1.hoc_label, prj1.connections.index(connection_id))))
            #w4.append(HocToPy.get('%s.object(%d).weight' % (prj2.hoc_label, prj1.connections.index(connection_id))))
            w3.append(hoc_list1.object(prj1.connections.index(connection_id)).weight[0])
            w4.append(hoc_list2.object(prj2.connections.index(connection_id)).weight[0])
        self.assertNotEqual(w1, w3) and self.assertNotEqual(w2, w4) 
        
    def testRandomizeDelays(self):
        # The probability of having two consecutive delays vector that are equal should be 0
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj2 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.randomizeDelays(self.distrib_Numpy)
        prj2.randomizeDelays(self.distrib_Native)
        d1 = []; d2 = []; d3 = []; d4 = []
        hoc_list1 = getattr(h, prj1.hoc_label)
        hoc_list2 = getattr(h, prj2.hoc_label)
        for connection_id in prj1.connections:
            #d1.append(HocToPy.get('%s.object(%d).delay' % (prj1.hoc_label, prj1.connections.index(connection_id))))
            #d2.append(HocToPy.get('%s.object(%d).delay' % (prj2.hoc_label, prj1.connections.index(connection_id))))
            d1.append(hoc_list1.object(prj1.connections.index(connection_id)).delay)
            d2.append(hoc_list2.object(prj2.connections.index(connection_id)).delay)
        prj1.randomizeDelays(self.distrib_Numpy)
        prj2.randomizeDelays(self.distrib_Native)
        for connection_id in prj1.connections:
            #d3.append(HocToPy.get('%s.object(%d).delay' % (prj1.hoc_label, prj1.connections.index(connection_id))))
            #d4.append(HocToPy.get('%s.object(%d).delay' % (prj2.hoc_label, prj1.connections.index(connection_id))))
            d3.append(hoc_list1.object(prj1.connections.index(connection_id)).delay)
            d4.append(hoc_list2.object(prj2.connections.index(connection_id)).delay)
        self.assertNotEqual(d1, d3) and self.assertNotEqual(d2, d4) 
               
        
    # If STDP works, a strong stimulation with only LTP should increase the mean weight
    def testSTDP(self):
        self.target.record()
        self.source.record()
        self.target.record_v()
        stdp_model = neuron.STDPMechanism(
            timing_dependence=neuron.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
            weight_dependence=neuron.AdditiveWeightDependence(w_min=0, w_max=2.0,
                                                              A_plus=0.1, A_minus=0.0),
            dendritic_delay_fraction=0.0
        )
        connector = neuron.AllToAllConnector(weights=1.0, delays=2.0)
        prj1 = neuron.Projection(self.source, self.target, connector,
                                 synapse_dynamics=neuron.SynapseDynamics(slow=stdp_model))
        #prj1.setDelays(2)
        #prj1.setWeights(1.0)
        mean_weight_before = 0
        hoc_list = getattr(h, prj1.hoc_label)
        for connection_id in prj1.connections:
            mean_weight_before += hoc_list.object(prj1.connections.index(connection_id)).weight[0]
        mean_weight_before = float(mean_weight_before/len(prj1.connections))  
        simtime = 100
        neuron.running = False
        run(simtime)
        mean_weight_after = 0
        self.target.print_v("tmp.v")
        source_spikes = self.source.meanSpikeCount()
        target_spikes = self.target.meanSpikeCount()
        if neuron.myid == 0:
            assert source_spikes > 0
            assert target_spikes > 0
        for connection_id in prj1.connections:
            mean_weight_after += hoc_list.object(prj1.connections.index(connection_id)).weight[0]
        mean_weight_after = float(mean_weight_after/len(prj1.connections))
        assert (mean_weight_before < mean_weight_after), "%g !< %g" % (mean_weight_before, mean_weight_after)
    
    def testSetDelaysWithSTDP(self):
        stdp_model = neuron.STDPMechanism(
            timing_dependence=neuron.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
            weight_dependence=neuron.AdditiveWeightDependence(w_min=0, w_max=2.0,
                                                              A_plus=0.1, A_minus=0.0),
            dendritic_delay_fraction=0.0
        )
        connector = neuron.AllToAllConnector(weights=1.0, delays=2.0)
        prj1 = neuron.Projection(self.source, self.target, connector,
                                 synapse_dynamics=neuron.SynapseDynamics(slow=stdp_model))
        pre2wa = getattr(h, '%s_pre2wa'  % prj1.hoc_label)
        post2wa = getattr(h, '%s_post2wa'  % prj1.hoc_label)
        assert pre2wa[0].delay == 2.0
        assert post2wa[0].delay == 0.0
        prj1.setDelays(3.0)
        assert pre2wa[0].delay == 3.0
        assert post2wa[0].delay == 0.0
    
    def testRandomizeDelaysWithSTDP(self):
        stdp_model = neuron.STDPMechanism(
            timing_dependence=neuron.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
            weight_dependence=neuron.AdditiveWeightDependence(w_min=0, w_max=2.0,
                                                              A_plus=0.1, A_minus=0.0),
            dendritic_delay_fraction=0.0
        )
        connector = neuron.AllToAllConnector(weights=1.0, delays=2.0)
        prj1 = neuron.Projection(self.source, self.target, connector,
                                 synapse_dynamics=neuron.SynapseDynamics(slow=stdp_model))
        pre2wa = getattr(h, '%s_pre2wa'  % prj1.hoc_label)
        post2wa = getattr(h, '%s_post2wa'  % prj1.hoc_label)
        nc_list = getattr(h, prj1.hoc_label)
        assert pre2wa[0].delay == 2.0
        assert post2wa[0].delay == 0.0
        prj1.randomizeDelays(self.distrib_Native)
        assert pre2wa[0].delay != 2.0 # check it is equal to the actual synaptic delay
        assert post2wa[0].delay == 0.0
        delay0 = nc_list.object(0).delay
        assert pre2wa[0].delay == delay0
        prj1.randomizeDelays(self.distrib_Numpy)
        assert pre2wa[0].delay != delay0 # check it is equal to the actual synaptic delay
        assert post2wa[0].delay == 0.0
        assert pre2wa[0].delay == nc_list.object(0).delay
    
    def testSetTopographicDelay(self):
        # We fix arbitrarily the positions of 2 cells in 2 populations and check 
        # the topographical delay between them is linked to the distance
        self.source[0,0].position = (0,0,0)
        self.target[2,2].position = (0,10,0)
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        rule="5.432*d"
        prj1.setTopographicDelays(rule)
        hoc_list = getattr(h, prj1.hoc_label)
        for connection_id in range(len(prj1.connections)):
            src = prj1.connections[connection_id][0]
            tgt = prj1.connections[connection_id][1]
            if (src == self.source[0,0]) and (tgt == self.target[2,2]):
                delay = hoc_list.object(prj1.connections.index(prj1.connections[connection_id])).delay
                assert (delay == 54.32), delay

class IDTest(unittest.TestCase):
    """Tests of the ID class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.pop1 = neuron.Population((5,), neuron.IF_curr_alpha,{'tau_m':10.0})
        self.pop2 = neuron.Population((5,4), neuron.IF_curr_exp,{'v_reset':-60.0})
    
    def testIDSetAndGet(self):
        if self.pop1[3] in self.pop1.gidlist:
            self.pop1[3].tau_m = 20.0
            self.assertEqual(20.0, self.pop1[3].tau_m)
        if self.pop1[0] in self.pop1.gidlist:
            self.assertEqual(10.0, self.pop1[0].tau_m)
        if self.pop2[3,2] in self.pop2.gidlist:
            self.pop2[3,2].v_reset = -70.0
            self.assertEqual(-70.0, self.pop2[3,2].v_reset)
        if self.pop2[0,0] in self.pop2.gidlist:
            self.assertEqual(-60.0, self.pop2[0,0].v_reset)

    def testGetCellClass(self):
        self.assertEqual(neuron.IF_curr_alpha, self.pop1[0].cellclass)
        self.assertEqual(neuron.IF_curr_exp, self.pop2[4,3].cellclass)
        
    def testSetAndGetPosition(self):
        self.assert_((self.pop2[0,2].position == (0.0,2.0,0.0)).all())
        new_pos = (0.5,1.5,0.0)
        self.pop2[0,2].position = new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())
        new_pos = (-0.6,3.5,-100.0) # check that position is set-by-value from new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())
        
if __name__ == "__main__":
    #sys.argv = ['./nrnpython']
    if '-python' in sys.argv:
        sys.argv.remove('-python')
    for arg in sys.argv:
        if 'bin/nrniv' in arg:
            sys.argv.remove(arg)
    #print sys.argv
    neuron.setup()
    unittest.main()
    neuron.end()
    