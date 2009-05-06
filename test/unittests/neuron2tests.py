"""
Unit tests for pyNN/neuron.py.
$Id: neurontests.py 364 2008-06-13 15:07:23Z apdavison $
"""

import pyNN.neuron as neuron
import pyNN.common as common
import pyNN.random as random
import unittest, sys, numpy
import numpy.random
import logging

def is_NetCon(obj):
    return hasattr(obj, 'weight')

def is_local(id):
    return isinstance(id, neuron.simulator.ID)

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def tearDown(self):
        neuron.simulator.state.gid_counter = 0
    
    def testCreateStandardCell(self):
        logging.info('=== CreationTest.testCreateStandardCell() ===')
        ifcell = neuron.create(neuron.IF_curr_alpha)
        assert ifcell == 0, 'Failed to create standard cell (cell=%s)' % ifcell
        ss = neuron.create(neuron.SpikeSourceArray)
        assert ss == 1, 'Failed to create standard cell (cell=%s)' % ss
        aeifcell = neuron.create(neuron.EIF_cond_exp_isfa_ista)
        assert aeifcell == 2, 'Failed to create standard cell (cell=%s)' % aeifcell
        hardwarecell = neuron.create(neuron.IF_facets_hardware1)
        assert hardwarecell == 3, 'Failed to create standard cell (cell=%s)' % hardwarecell
        
    def testCreateStandardCells(self):
        """create(): Creating multiple cells should return a list of integers"""
        logging.info('=== CreationTest.testCreateStandardCells ===')
        ifcells = neuron.create(neuron.IF_curr_alpha, n=10)
        assert ifcells == range(0,10), 'Failed to create 10 standard cells'
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        logging.info('=== CreationTest.testCreateStandardCellsWithNegative_n ===')
        self.assertRaises(AssertionError, neuron.create, neuron.IF_curr_alpha, n=-1)
       
    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with the top-level HocObject"""
        logging.info('=== CreationTest.testCreateStandardCellWithParams ===')
        ifcell = neuron.create(neuron.IF_curr_alpha,{'tau_syn_E':3.141592654})
        #self.assertAlmostEqual(HocToPy.get('cell%d.esyn.tau' % ifcell, 'float'), 3.141592654, places=5)
        try:
            self.assertAlmostEqual(getattr(neuron.h, 'cell%d' % ifcell).esyn.tau, 3.141592654, places=5)
        except AttributeError: # if the cell is not on that node
            pass
        
    def testCreateNEURONCell(self):
        """create(): First cell created should have index 0."""
        logging.info('=== CreationTest.testCreateNEURONCell ===')
        ifcell = neuron.create(neuron.StandardIF, {'syn_type':'current','syn_shape':'exp'})
        assert ifcell == 0, 'Failed to create NEURON-specific cell'
    
    def testCreateInvalidCell(self):
        """create(): Trying to create a cell type which is not a standard cell or
        valid native cell should raise a HocError."""
        self.assertRaises(common.InvalidModelError, neuron.create, 'qwerty', n=10)
    
    def testCreateWithInvalidParameter(self):
        """create(): Creating a cell with an invalid parameter should raise an Exception."""
        self.assertRaises(common.NonExistentParameterError, neuron.create, neuron.IF_curr_alpha, {'tau_foo':3.141592654})        


# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        logging.info("=== ConnectionTest.setUp() ===")
        self.postcells = neuron.create(neuron.IF_curr_alpha, n=3)
        self.precells = neuron.create(neuron.SpikeSourcePoisson, n=5)
        
    def tearDown(self):
        logging.info("=== ConnectionTest.tearDown() ===")
        neuron.gid_counter = 0
        
    def testConnectTwoCells(self):
        """connect(): The first connection created should have id 0."""
        logging.info("=== ConnectionTest.testConnectTwoCells ===")
        conn_list = neuron.connect(self.precells[0], self.postcells[0])
        # conn will be an empty list if it does not exist on that node
        self.assertEqual(len(conn_list), 1)
        assert is_NetCon(conn_list[0].nc), 'Error creating connection, conn_list=%s' % conn_list
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        logging.info("=== ConnectionTest.testConnectTwoCellsWithWeight() ===")
        conn_list = neuron.connect(self.precells[0], self.postcells[0], weight=0.1234)
        if conn_list:
            weight = conn_list[0].nc.weight[0]
            assert weight == 0.1234, "Weight set (0.1234) does not match weight retrieved (%s)" % weight
    
    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        conn_list = neuron.connect(self.precells[0], self.postcells[0], delay=4.321)
        if conn_list:
            delay = conn_list[0].nc.delay
            assert delay == 4.321, "Delay set (4.321) does not match delay retrieved (%s)." % delay
    
    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n,
        each element being the id number of a netcon."""
        conn_list = neuron.connect(self.precells, self.postcells[0])
        # connections are only created on the node containing the post-syn
        self.assertEqual(len(conn_list), len(self.precells))
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        conn_list = neuron.connect(self.precells[0], self.postcells)
        cells_on_this_node = len([i for i in self.postcells if is_local(i)])
        self.assertEqual(len(conn_list),  cells_on_this_node)
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        conn_list = neuron.connect(self.precells, self.postcells)
        cells_on_this_node = len([i for i in self.postcells if is_local(i)])
        expected_length = cells_on_this_node*len(self.precells)
        self.assertEqual(len(conn_list), expected_length, "%d != %d" % (len(conn_list), expected_length))
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        conn_list = neuron.connect(self.precells, self.postcells, p=0.5)
        cells_on_this_node = len([i for i in self.postcells if is_local(i)])
        assert 0 < len(conn_list) < len(self.precells)*cells_on_this_node, 'Number of connections is %d: this is very unlikely (although possible).' % len(conn_list)
    
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
        self.postcells.append('99.9')
        self.assertRaises(common.ConnectionError, neuron.connect, self.precells, self.postcells)

    def testConnectTooSmallDelay(self):
        self.assertRaises(common.ConnectionError, neuron.connect, self.precells[0], self.postcells[0], delay=1e-12)

# ==============================================================================
class SetValueTest(unittest.TestCase):
    
    def setUp(self):
        self.cells = neuron.create(neuron.IF_curr_exp, n=10)
        self.native_cells = neuron.create(neuron.StandardIF, dict(syn_shape="exp", syn_type="current"), n=2)
        self.single_cell = neuron.create(neuron.IF_cond_exp, n=1)
        
    def testSetFloat(self):
        logging.info("=== SetValueTest.testSetFloat() ===")
        neuron.set(self.cells, 'tau_m',35.7)
        neuron.set(self.native_cells, 'cm', 0.987)
        neuron.set(self.single_cell, 'v_init', -67.8)
        for cell in self.cells:
            try:
                assert cell.tau_m == 35.7
            except AttributeError: # if cell is not on this node
                pass
        for cell in self.native_cells:
            assert cell._cell.seg.cm == 0.987
        assert self.single_cell.v_init == -67.8
  
#    #def testSetString(self):
#    #    neuron.set(self.cells, neuron.IF_curr_exp,'param_name','string_value')
#    ## note we don't currently have any models with string parameters, so
#    ## this is all commented out
#    #    for cell in self.cells:
#    #        assert HocToPy.get('cell%d.param_name' % cell, 'string') == 'string_value'

    def testSetDict(self):
        neuron.set(self.cells, {'tau_m': 35.7, 'tau_syn_E': 5.432})
        for cell in self.cells:
            try:
                assert cell._cell.tau_e == 5.432
                assert cell.tau_m == 35.7
            except AttributeError: # if cell is not on this node
                pass
            
    def testSetNonExistentParameter(self):
        # note that although syn_shape is added to the parameter dict when creating
        # an IF_curr_exp, it is not a valid parameter to be changed later.
        self.assertRaises(common.NonExistentParameterError, neuron.set, self.cells, 'syn_shape', 'alpha')

    def testMembInit(self):
        native_cell = self.cells[0]._cell
        a = native_cell.v_init
        native_cell.memb_init(a+1.0)
        self.assertEqual(native_cell.v_init, a+1.0)
        self.assertEqual(native_cell.v_init, native_cell.seg.v)
        
# ==============================================================================
class RecordTest(unittest.TestCase):

    def setUp(self):
        self.cells = neuron.create(neuron.IF_curr_exp, n=10)
        self.native_cells = neuron.create(neuron.StandardIF, dict(syn_shape="exp", syn_type="current"), n=2)
        self.single_cell = neuron.create(neuron.IF_cond_exp, n=1)

    def testRecordSpikes(self):
        neuron.record(self.cells, "tmp.spk")
        neuron.record(self.native_cells, "tmp1.spk")
        neuron.record(self.single_cell, "tmp2.spk")
        # need a better test

    def testRecordV(self):
        neuron.record_v(self.cells, "tmp.v")
        neuron.record_v(self.native_cells, "tmp1.v")
        neuron.record_v(self.single_cell, "tmp2.v")

    def testStopRecording(self):
        self.cells[0]._cell.record_v(0)
        self.native_cells[0]._cell.record_v(0)
        self.assertEqual(self.cells[0]._cell.vtrace, None)
        self.assertEqual(self.native_cells[0]._cell.vtrace, None)
        
    def testRecordGSyn(self):
        self.cells[0]._cell.record_gsyn('esyn', 1)
        assert hasattr(self.gsyn_trace['esyn'], 'size') # hoc Vector object

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
        self.assertEqual(net.size, 9)
        n_cells_local = len([id for id in net])
        # round-robin distribution
        min = 9/neuron.num_processes()
        max = min+1
        assert min <= n_cells_local <= max, "%d not between %d and %d" % (n_cells_local, min, max)
    
    def testInitWithParams(self):
        """Population.__init__(): Parameters set on creation should be the same as
        retrieved with the top-level HocObject"""
        net = neuron.Population((3,3), neuron.IF_curr_alpha, {'tau_syn_E':3.141592654})
        for cell in net:
            tau_syn = cell._cell.esyn.tau
            self.assertAlmostEqual(tau_syn, 3.141592654, places=5)
    
    def testInitWithLabel(self):
        """Population.__init__(): A label set on initialisation should be retrievable with the Population.label attribute."""
        net = neuron.Population((3,3), neuron.IF_curr_alpha, label='iurghiushrg')
        assert net.label == 'iurghiushrg'
    
    def testInvalidCellType(self):
        """Population.__init__(): Trying to create a cell type which is not a StandardCell
        or a valid neuron model should raise a HocError."""
        self.assertRaises(common.InvalidModelError, neuron.Population, (3,3), 'qwerty', {})
        
    def testInitWithNonStandardModel(self):
        """Population.__init__(): the cell list in hoc should have the same length as the population size."""
        net = neuron.Population((3,3), neuron.StandardIF, {'syn_type':'current', 'syn_shape':'exp'})
        self.assertEqual(net.size, 9)
        n_cells_local = len([id for id in net])
        min = 9/neuron.num_processes()
        max = min+1
        assert min <= n_cells_local <= max, "%d not between %d and %d" % (n_cells_local, min, max)

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
            self.assertEqual(ids, net.local_cells.tolist())
            self.assert_(isinstance(ids[0], neuron.simulator.ID))
            
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
        self.net2 = neuron.Population((5,), neuron.StandardIF, {'syn_type':'current','syn_shape':'exp'})
    
    def testSetFromDict(self):
        """Population.set()"""
        self.net.set({'tau_m': 43.21})
        for cell in self.net:
            assert cell.tau_m == 43.21
    
    def testSetFromPair(self):
        """Population.set(): A parameter set as a string, value pair should be retrievable using the top-level HocObject"""
        self.net.set('tau_m', 12.34)
        for cell in self.net:
            assert cell.tau_m == 12.34
    
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
        for cell in self.net2:
            assert cell.tau_m == 43.21
        
    def testTSet(self):
        """Population.tset(): The valueArray passed should be retrievable using the top-level HocObject on all nodes."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.net.tset('i_offset', array_in)
        array_out = self.net.get('i_offset', as_array=True)
        array_out2 = numpy.zeros((3,3), float)
        for i in 0,1,2:
            for j in 0,1,2:
                id = 3*i+j
                cell = self.net[i,j]
                if cell in self.net:
                    array_out2[i, j] = cell._cell.stim.amp
                else:
                    array_out2[i, j] = array_in[i, j]
        assert numpy.equal(array_in, array_out, array_out2).all(), array_out
    
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
        self.assertNotEqual(self.net[0,0].tau_m,
                            self.net[0,1].tau_m)
        
    def testRSetNumpy(self):
        """Population.rset(): with numpy rng."""
        rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.net.rset('cm', rd1)
        output_values = self.net.get('cm', as_array=True)
        input_values = rd2.next(9).reshape(self.net.dim)
        assert numpy.equal(input_values, output_values).all()
        
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
        for i in 0,1,2:
            for j in 0,1,2:
                id = self.net[i,j]
                output_values_1[i, j] = id._cell(0.5).cm
        self.net.rset('cm', rd2)
        output_values_2 = self.net.get('cm', as_array=True)
        assert numpy.equal(output_values_1, output_values_2).all()

# ==============================================================================
class PopulationRecordTest(unittest.TestCase): # to write later
    """Tests of the record(), record_v(), printSpikes(), print_v() and
       meanSpikeCount() methods of the Population class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.pop1 = neuron.Population((3,3), neuron.SpikeSourcePoisson,{'rate': 20})
        self.pop2 = neuron.Population((3,3), neuron.IF_curr_alpha)
        self.pop3 = neuron.Population((3,3), neuron.EIF_cond_alpha_isfa_ista)
        self.pops =[self.pop1, self.pop2, self.pop3]

    def tearDown(self):
        neuron.simulator.reset()

    def testRecordAll(self):
        """Population.record(): not a full test, just checking there are no Exceptions raised."""
        for pop in self.pops:
            pop.record()
        
    def testRecordInt(self):
        """Population.record(n): not a full test, just checking there are no Exceptions raised."""
        # Partial record
        for pop in self.pops:
            pop.record(5)

    def testRecordWithRNG(self):
        """Population.record(n, rng): not a full test, just checking there are no Exceptions raised."""
        for pop in self.pops:
            pop.record(5, random.NumpyRNG())

    def testRecordList(self):
        """Population.record(list): not a full test, just checking there are no Exceptions raised."""
        # Selected list record
        record_list = []
        for pop in self.pops:
            for i in range(0,2):
                record_list.append(pop[i,1])
            pop.record(record_list)

    def testInvalidCellList(self):
        self.assertRaises(Exception, self.pop2.record, 4.2)

    def testSpikeRecording(self):
        # We test the mean spike count by checking if the rate of the poissonian sources are
        # close to 20 Hz. Then we also test how the spikes are saved
        self.pop1.record()
        self.pop3.record()
        simtime = 1000.0
        neuron.run(simtime)
        #self.pop1.printSpikes("temp_neuron.ras", gather=True)
        rate = self.pop1.meanSpikeCount()*1000/simtime
        if neuron.rank() == 0: # only on master node
            assert (20*0.8 < rate) and (rate < 20*1.2), "rate is %s" % rate
        rate = self.pop3.meanSpikeCount()*1000/simtime
        self.assertEqual(rate, 0.0)

    def testPotentialRecording(self):
        """Population.record_v() and Population.print_v(): not a full test, just checking 
        # there are no Exceptions raised."""
        rng = random.NumpyRNG(123)
        v_reset  = -65.0
        v_thresh = -50.0
        uniformDistr = random.RandomDistribution(rng=rng, distribution='uniform', parameters=[v_reset, v_thresh])
        self.pop2.randomInit(uniformDistr)
        self.pop2.record_v([self.pop2[0,0], self.pop2[1,1]])
        simtime = 10.0
        neuron.running = False
        neuron.run(simtime)
        self.pop2.print_v("temp_neuron.v", gather=True, compatible_output=True)

    def testRecordWithSpikeTimesGreaterThanSimTime(self):
        """
        If a `SpikeSourceArray` is initialized with spike times greater than the
        simulation time, only those spikes that actually occurred should be
        written to file or returned by getSpikes().
        """
        spike_times = numpy.arange(10.0, 200.0, 10.0)
        spike_source = neuron.Population(1, neuron.SpikeSourceArray, {'spike_times': spike_times})
        spike_source.record()
        neuron.run(100.0)
        spikes = spike_source.getSpikes()
        spikes = spikes[:,1]
        if neuron.rank() == 0:
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
        self.target1     = neuron.Population((1,), neuron.IF_cond_exp)
        self.source5     = neuron.Population((5,), neuron.SpikeSourcePoisson)
        self.source22    = neuron.Population((2,2), neuron.SpikeSourcePoisson)
        self.source33    = neuron.Population((3,3), neuron.SpikeSourcePoisson)
        self.expoisson33 = neuron.Population((3,3), neuron.SpikeSourcePoisson,{'rate': 100})
        
    def testAllToAll(self):
        """For all connections created with "allToAll" it should be possible to
        obtain the weight using the top-level HocObject"""
        for srcP in [self.source5, self.source22, self.target33]:
            for tgtP in [self.target6, self.target33]:
                #print "gid_counter = ", neuron.simulator.state.gid_counter
                if srcP == tgtP:
                    prj = neuron.Projection(srcP, tgtP, neuron.AllToAllConnector(allow_self_connections=False))
                else:
                    prj = neuron.Projection(srcP, tgtP, neuron.AllToAllConnector())
                prj.setWeights(1.234)
                weights = []
                for c in prj.connections:
                    weights.append(c.nc.weight[0])
                assert weights == [1.234]*len(prj)
        
            
    def testFixedProbability(self):
        """For all connections created with "fixedProbability"..."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target1, self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, neuron.FixedProbabilityConnector(0.5), rng=random.NumpyRNG(12345))
                prj2 = neuron.Projection(srcP, tgtP, neuron.FixedProbabilityConnector(0.5), rng=random.NativeRNG(12345))
                for prj in prj1, prj2:
                    assert (0 < len(prj) < len(srcP)*len(tgtP)), 'len(prj) = %d, len(srcP)*len(tgtP) = %d' % (len(prj), len(srcP)*len(tgtP))
                
    def testOneToOne(self):
        """For all connections created with "OneToOne" ..."""
        prj = neuron.Projection(self.source33, self.target33, neuron.OneToOneConnector())
        assert len(prj.connections) == len(self.target33.local_cells), prj.connections
     
    def testDistanceDependentProbability(self):
        """For all connections created with "distanceDependentProbability"..."""
        # Test should be improved..."
        for rngclass in (random.NumpyRNG, random.NativeRNG):
            for expr in ('exp(-d)', 'd < 0.5'):
        #rngclass = random.NumpyRNG
        #expr = 'exp(-d)'
                prj = neuron.Projection(self.source33, self.target33,
                                        neuron.DistanceDependentProbabilityConnector(d_expression=expr),
                                        rng=rngclass(12345))
                assert (0 < len(prj) < len(self.source33)*len(self.target33))
        self.assertRaises(ZeroDivisionError, neuron.DistanceDependentProbabilityConnector, d_expression="d/0.0")

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
        self.assertRaises(Exception, neuron.FixedNumberPreConnector, None)
        

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
        self.assertRaises(Exception, neuron.FixedNumberPostConnector, None)

    def testFromList(self):
        c1 = neuron.FromListConnector([
            ([0,], [0,], 0.1, 0.1),
            ([3,], [0,], 0.2, 0.11),
            ([2,], [3,], 0.3, 0.12),
            ([4,], [2,], 0.4, 0.13),
            ([0,], [1,], 0.5, 0.14),
            ])
        prj = neuron.Projection(self.source5, self.target6, c1)
        self.assertEqual(len(prj.connections), 5)
            
            

    def testSaveAndLoad(self):
        prj1 = neuron.Projection(self.source33, self.target33, neuron.OneToOneConnector())
        prj1.setDelays(1)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp", gather=False)
        if neuron.num_processes() > 1:
            distributed = True
        else:
            distributed = False
        prj2 = neuron.Projection(self.source33, self.target33, neuron.FromFileConnector("connections.tmp",
                                                                                        distributed=distributed))
        w1 = []; w2 = []; d1 = []; d2 = []
        # For a connections scheme saved and reloaded, we test if the connections, their weights and their delays
        # are equal.
        for c1,c2 in zip(prj1.connections, prj2.connections):
            w1.append(c1.nc.weight[0])
            w2.append(c2.nc.weight[0])
            d1.append(c1.nc.delay)
            d2.append(c2.nc.delay)
        assert (w1 == w2), 'w1 = %s\nw2 = %s' % (w1, w2)
        assert (d1 == d2), 'd1 = %s\nd2 = %s' % (d1, d2)
          
    def testSettingDelays(self):
        """Delays should be set correctly when using a Connector object."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, neuron.AllToAllConnector(delays=0.321))
                assert prj1.connections[0].nc.delay == 0.321, "Delay should be 0.321, actually %g" % prj1.connections[0].nc.delay

class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), randomizeWeights() and
    randomizeDelays() methods of the Projection class."""

    def setUp(self):
        self.target   = neuron.Population((3,3), neuron.IF_curr_alpha)
        self.source   = neuron.Population((3,3), neuron.SpikeSourcePoisson,{'rate': 200})
        self.distrib_Numpy = random.RandomDistribution(rng=random.NumpyRNG(12345), distribution='uniform', parameters=(0.2,1)) 
        self.distrib_Native= random.RandomDistribution(rng=random.NativeRNG(12345), distribution='uniform', parameters=(0.2,1)) 
        
    def testSetWeights(self):
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.setWeights(2.345)
        weights = []
        for c in prj1.connections:
            weights.append(c.nc.weight[0])
        result = 2.345*numpy.ones(len(prj1.connections))
        assert (weights == result.tolist())
        
    def testSetDelays(self):
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.setDelays(2.345)
        delays = []
        for c in prj1.connections:
            delays.append(c.nc.delay)
        result = 2.345*numpy.ones(len(prj1.connections))
        assert (delays == result.tolist())
        
    def testRandomizeWeights(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj2 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        w1 = []; w2 = []; w3 = []; w4 = []
        for c in prj1.connections:
            w1.append(c.nc.weight[0])
            w2.append(c.nc.weight[0])
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        for c in prj1.connections:
            w3.append(c.nc.weight[0])
            w4.append(c.nc.weight[0])
        self.assertNotEqual(w1, w3) and self.assertNotEqual(w2, w4) 
        
    def testRandomizeDelays(self):
        # The probability of having two consecutive delays vector that are equal should be 0
        prj1 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj2 = neuron.Projection(self.source, self.target, neuron.AllToAllConnector())
        prj1.randomizeDelays(self.distrib_Numpy)
        prj2.randomizeDelays(self.distrib_Native)
        d1 = []; d2 = []; d3 = []; d4 = []
        for c in prj1.connections:
            d1.append(c.nc.weight[0])
            d2.append(c.nc.weight[0])
        prj1.randomizeWeights(self.distrib_Numpy)
        prj2.randomizeWeights(self.distrib_Native)
        for c in prj1.connections:
            d3.append(c.nc.weight[0])
            d4.append(c.nc.weight[0])
        self.assertNotEqual(d1, d3) and self.assertNotEqual(d2, d4) 
               
        
#    # If STDP works, a strong stimulation with only LTP should increase the mean weight
#    def testSTDP(self):
#        self.target.record()
#        self.source.record()
#        self.target.record_v()
#        stdp_model = neuron.STDPMechanism(
#            timing_dependence=neuron.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
#            weight_dependence=neuron.AdditiveWeightDependence(w_min=0, w_max=2.0,
#                                                              A_plus=0.1, A_minus=0.0),
#            dendritic_delay_fraction=0.0
#        )
#        connector = neuron.AllToAllConnector(weights=1.0, delays=2.0)
#        prj1 = neuron.Projection(self.source, self.target, connector,
#                                 synapse_dynamics=neuron.SynapseDynamics(slow=stdp_model))
#        #prj1.setDelays(2)
#        #prj1.setWeights(1.0)
#        mean_weight_before = 0
#        hoc_list = getattr(h, prj1.label)
#        for connection_id in prj1.connections:
#            mean_weight_before += hoc_list.object(prj1.connections.index(connection_id)).weight[0]
#        mean_weight_before = float(mean_weight_before/len(prj1.connections))  
#        simtime = 100
#        neuron.running = False
#        run(simtime)
#        mean_weight_after = 0
#        self.target.print_v("tmp.v")
#        source_spikes = self.source.meanSpikeCount()
#        target_spikes = self.target.meanSpikeCount()
#        if neuron.myid == 0:
#            assert source_spikes > 0
#            assert target_spikes > 0
#        for connection_id in prj1.connections:
#            mean_weight_after += hoc_list.object(prj1.connections.index(connection_id)).weight[0]
#        mean_weight_after = float(mean_weight_after/len(prj1.connections))
#        assert (mean_weight_before < mean_weight_after), "%g !< %g" % (mean_weight_before, mean_weight_after)
#    
#    def testSetDelaysWithSTDP(self):
#        stdp_model = neuron.STDPMechanism(
#            timing_dependence=neuron.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
#            weight_dependence=neuron.AdditiveWeightDependence(w_min=0, w_max=2.0,
#                                                              A_plus=0.1, A_minus=0.0),
#            dendritic_delay_fraction=0.0
#        )
#        connector = neuron.AllToAllConnector(weights=1.0, delays=2.0)
#        prj1 = neuron.Projection(self.source, self.target, connector,
#                                 synapse_dynamics=neuron.SynapseDynamics(slow=stdp_model))
#        pre2wa = getattr(h, '%s_pre2wa'  % prj1.hoc_label)
#        post2wa = getattr(h, '%s_post2wa'  % prj1.hoc_label)
#        assert pre2wa[0].delay == 2.0
#        assert post2wa[0].delay == 0.0
#        prj1.setDelays(3.0)
#        assert pre2wa[0].delay == 3.0
#        assert post2wa[0].delay == 0.0
#    
#    def testRandomizeDelaysWithSTDP(self):
#        stdp_model = neuron.STDPMechanism(
#            timing_dependence=neuron.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
#            weight_dependence=neuron.AdditiveWeightDependence(w_min=0, w_max=2.0,
#                                                              A_plus=0.1, A_minus=0.0),
#            dendritic_delay_fraction=0.0
#        )
#        connector = neuron.AllToAllConnector(weights=1.0, delays=2.0)
#        prj1 = neuron.Projection(self.source, self.target, connector,
#                                 synapse_dynamics=neuron.SynapseDynamics(slow=stdp_model))
#        pre2wa = getattr(h, '%s_pre2wa'  % prj1.hoc_label)
#        post2wa = getattr(h, '%s_post2wa'  % prj1.hoc_label)
#        nc_list = getattr(h, prj1.label)
#        assert pre2wa[0].delay == 2.0
#        assert post2wa[0].delay == 0.0
#        prj1.randomizeDelays(self.distrib_Native)
#        assert pre2wa[0].delay != 2.0 # check it is equal to the actual synaptic delay
#        assert post2wa[0].delay == 0.0
#        delay0 = nc_list.object(0).delay
#        assert pre2wa[0].delay == delay0
#        prj1.randomizeDelays(self.distrib_Numpy)
#        assert pre2wa[0].delay != delay0 # check it is equal to the actual synaptic delay
#        assert post2wa[0].delay == 0.0
#        assert pre2wa[0].delay == nc_list.object(0).delay
#    
#    def testSetTopographicDelay(self):
#        # We fix arbitrarily the positions of 2 cells in 2 populations and check 
#        # the topographical delay between them is linked to the distance
#        self.source[0,0].position = (0,0,0)
#        self.target[2,2].position = (0,10,0)
#        prj1 = neuron.Projection(self.source, self.target, 'allToAll')
#        rule="5.432*d"
#        prj1.setTopographicDelays(rule)
#        hoc_list = getattr(h, prj1.label)
#        for connection_id in range(len(prj1.connections)):
#            src = prj1.connections[connection_id][0]
#            tgt = prj1.connections[connection_id][1]
#            if (src == self.source[0,0]) and (tgt == self.target[2,2]):
#                delay = hoc_list.object(prj1.connections.index(prj1.connections[connection_id])).delay
#                assert (delay == 54.32), delay
#
class IDTest(unittest.TestCase):
    """Tests of the ID class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.pop1 = neuron.Population((5,), neuron.IF_curr_alpha,{'tau_m':10.0})
        self.pop2 = neuron.Population((5,4), neuron.IF_curr_exp,{'v_reset':-60.0})
    
    def testIDSetAndGet(self):
        if self.pop1[3] in self.pop1:
            self.pop1[3].tau_m = 20.0
            self.assertEqual(20.0, self.pop1[3].tau_m)
        if self.pop1[0] in self.pop1:
            self.assertEqual(10.0, self.pop1[0].tau_m)
        if self.pop2[3,2] in self.pop2:
            self.pop2[3,2].v_reset = -70.0
            self.assertEqual(-70.0, self.pop2[3,2].v_reset)
        if self.pop2[0,0] in self.pop2:
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
        
class SynapticPlasticityTest(unittest.TestCase):
    
    def setUp(self):
        neuron.Population.nPop = 0
        neuron.Projection.nProj = 0
        self.target33    = neuron.Population((3,3), neuron.IF_curr_alpha)
        self.target6     = neuron.Population((6,), neuron.IF_cond_exp)
        self.source5     = neuron.Population((5,), neuron.IF_curr_exp)
        self.source22    = neuron.Population((2,2), neuron.SpikeSourcePoisson)
        
    def testUseTsodyksMarkram(self):
        U=0.6
        tau_rec=60.0
        tau_facil=6.0
        u0=0.6
        self.assertRaises(Exception,
                          self.target33[0,0]._cell.use_Tsodyks_Markram_synapses,
                          'excitatory', U, tau_rec, tau_facil, u0)
        native_cell = self.target6[0]._cell
        native_cell.use_Tsodyks_Markram_synapses(
            'excitatory', U, tau_rec, tau_facil, u0)
        self.assertEqual(native_cell.esyn.tau, native_cell.tau_e)
        self.assertEqual(native_cell.esyn.e, native_cell.e_e)
        self.assertEqual(native_cell.esyn.U, U)
        self.assertRaises(AttributeError, lambda x: native_cell.isyn.U, None)
        native_cell.use_Tsodyks_Markram_synapses(
            'inhibitory', U, tau_rec, tau_facil, u0)
        self.assertEqual(native_cell.isyn.tau, native_cell.tau_i)
        self.assertEqual(native_cell.isyn.e, native_cell.e_i)
        self.assertEqual(native_cell.isyn.U, U)

class LoadMechanismsTest(unittest.TestCase):
    
    def test_load_mechanisms(self):
        self.assertRaises(Exception, neuron.simulator.load_mechanisms, path="/dev/null")    

        
class SetupTest(unittest.TestCase):
    
    def test_cvode(self):
        neuron.setup(use_cvode=True)
        
        
if __name__ == "__main__":
    if '-python' in sys.argv:
        sys.argv.remove('-python')
    for arg in sys.argv:
        if 'bin/nrniv' in arg:
            sys.argv.remove(arg)
    neuron.setup()
    unittest.main()
    neuron.end()
    