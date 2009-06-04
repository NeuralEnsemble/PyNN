"""
Unit tests for pyNN.nest module
$Id:nesttests.py 5 2007-04-16 15:01:24Z davison $
"""

import pyNN.nest as nest
import pyNN.common as common
import pyNN.random as random
import unittest
import numpy
import os

nest.connectors.CHECK_CONNECTIONS = True

def get_weight(src, port, plasticity_name):
    conn_dict = nest.nest.GetConnection([src], plasticity_name, port)
    if isinstance(conn_dict, dict):
        return conn_dict['weight']
    else:
        raise Exception("Either the source id (%s) or the port number (%s) or both is invalid." % (src, port))

def get_delay(src, port, plasticity_name):
    conn_dict = nest.nest.GetConnection([src], plasticity_name, port)
    if isinstance(conn_dict, dict):
        return conn_dict['delay']
    else:
        raise Exception("Either the source id (%s) or the port number (%s) or both is invalid." % (src, port))

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) < threshold).all()

def max_array_diff(a, b):
    return max(abs(a-b))

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def setUp(self):
        nest.setup()
    
    def tearDown(self):
        pass
    
    def testCreateStandardCell(self):
        """create(): First cell created should have GID==1"""
        for cell_type in nest.list_standard_models():
            ifcell = nest.create(cell_type)
            assert ifcell.cellclass == cell_type
        
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
        ifcell_params = nest.nest.GetStatus([ifcell])
        assert ifcell_params[0]['tau_syn_ex'] == 3.141592654
 
    def testCreateNESTCell(self):
        """create(): First cell created should have GID==1"""
        ifcell = nest.create('iaf_neuron')
        assert ifcell == 1, 'Failed to create NEST-specific cell'
    
    def testCreateNonExistentCell(self):
        """create(): Trying to create a cell type which is not a standard cell or
        a NEST cell should raise a SLIError."""
        self.assertRaises(Exception, nest.create, 'qwerty')
    
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
        
    #def testConnectTwoCells(self):
    #    """connect(): The first connection created should have id 0."""
    #    conn = nest.connect(self.precells[0],self.postcells[0])
    #    assert conn == 0, 'Error creating connection'
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        conn_id = nest.connect(self.precells[0],self.postcells[0],weight=0.1234)[0]
        weight = conn_id._get_weight()
        self.assertEqual(weight, 0.1234) # note that pyNN.nest uses nA for weights, whereas NEST uses pA

    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        conn_id = nest.connect(self.precells[0],self.postcells[0],delay=4.4)[0]
        delay = conn_id._get_delay()
        assert delay == 4.4, "Delay set (4.4) does not match delay retrieved (%g)." % delay # Note that delays are only stored to the precision of the timestep.

    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n, each element being the target port."""
        connlist = nest.connect(self.precells,self.postcells[0])
        assert len(connlist) == len(self.precells)
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        connlist = nest.connect(self.precells[0],self.postcells)
        assert len(connlist) == len(self.postcells)
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        connlist = nest.connect(self.precells,self.postcells)
        assert len(connlist) == len(self.precells)*len(self.postcells)
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        connlist = nest.connect(self.precells,self.postcells,p=0.5)
        assert 0 < len(connlist) < len(self.precells)*len(self.postcells), 'Number of connections is %d: this is very unlikely (although possible).' % len(connlist)
    
    def testConnectNonExistentPreCell(self):
        """connect(): Connecting from non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, nest.connect, [12345], self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        non_existent_cell = nest.simulator.ID(45678)
        non_existent_cell.cellclass = Exception
        self.assertRaises(common.ConnectionError, nest.connect, self.precells[0], [non_existent_cell])
        
    def testDelayTooSmall(self):
        """connect(): Setting a delay smaller than min_delay should raise an Exception.""" 
        self.assertRaises(common.ConnectionError, nest.connect, self.precells[0], self.postcells[0], delay=0.0)
           
    def testDelayTooLarge(self):
        """connect(): Setting a delay larger than max_delay should raise an Exception.""" 
        self.assertRaises(common.ConnectionError, nest.connect, self.precells[0], self.postcells[0], delay=1000.0)

    #def __del__(self):
    #    nest.end()

# ==============================================================================        
class SetValueTest(unittest.TestCase):
    
    def testSetNonExistentNativeParameter(self):
        ifcell = nest.create('iaf_neuron')
        def set_val(x):
            ifcell.foo = x
        self.assertRaises(nest.nest.NESTError, set_val, 3.2)

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
        ifcell_params = nest.nest.GetStatus([net.cell[0,0]])
        assert ifcell_params[0]['tau_syn_ex'] == 3.141592654
    
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
        self.net5 = nest.Population((3,3),nest.IF_cond_alpha)
    
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
        self.net5 = nest.Population((3,3),nest.IF_cond_alpha)
        
    def testIter(self):
        """This needs more thought for the distributed case."""
        for net in self.net1, self.net2:
            ids = [i for i in net]
            self.assertEqual(ids, net.cell.flatten().tolist())
            self.assert_(isinstance(ids[0], nest.simulator.ID))
            
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
        assert nest.nest.GetStatus([self.net.cell[0,0]])[0]['tau_m'] == 43.21
        assert nest.nest.GetStatus([self.net.cell[0,0]])[0]['C_m'] == 987.0 # pF
    
    def testSetFromPair(self):
        """A parameter set as a string,value pair should be retrievable using pynest.getDict()"""
        self.net.set('tau_m',12.34)
        assert nest.nest.GetStatus([self.net.cell[0,0]])[0]['tau_m'] == 12.34
    
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
        """Parameters set in a dict should all be retrievable using nest.getStatus()"""
        self.net2.set({'tau_m':43.21})
        assert nest.nest.GetStatus([self.net2.cell[0]])[0]['tau_m'] == 43.21
    
    def testTSet(self):
        """The valueArray passed should be retrievable using get() on all nodes."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.net.tset('cm', array_in)
        array_out = self.net.get('cm', as_array=True).reshape((3,3))
        assert numpy.equal(array_in, array_out).all()
    
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
        output_values = self.net.get('cm', as_array=True).flatten()
        input_values = rd2.next(9)
        self.assert_( arrays_almost_equal(input_values, output_values, 1e-6),
                     "%s != %s" % (input_values, output_values) )


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
        simtime = 1000
        nest.run(simtime)
        self.pop1.printSpikes("temp_nest.ras")
        rate = self.pop1.meanSpikeCount()*1000/simtime
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

    def testRecordWithSpikeTimesGreaterThanSimTime(self):
        """
        If a `SpikeSourceArray` is initialized with spike times greater than the
        simulation time, only those spikes that actually occurred should be
        written to file or returned by getSpikes().
        """
        spike_times = numpy.arange(10.0, 200.0, 10.0)
        spike_source = nest.Population(1, nest.SpikeSourceArray, {'spike_times': spike_times})
        spike_source.record()
        nest.run(101.0)
        spikes = spike_source.getSpikes()[:,1]
        self.assert_( max(spikes) == 100.0, "max(spikes)=%s" % max(spikes) )

# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Projection class."""
        
    def setUp(self):
        nest.setup(max_delay=0.5)
        nest.Population.nPop = 0
        self.target33 = nest.Population((3,3),nest.IF_curr_alpha, label="target33")
        self.target6  = nest.Population((6,),nest.IF_cond_exp, label="target6")
        self.source5  = nest.Population((5,),nest.SpikeSourcePoisson, label="source5")
        self.source6  = self.target6
        self.source22 = nest.Population((2,2),nest.SpikeSourcePoisson, label="source22")
        self.source33 = nest.Population((3,3),nest.SpikeSourcePoisson, label="source33")

    def testAllToAll(self):
        """For all connections created with "allToAll" it should be possible to obtain the weight using pynest.getWeight()"""
        for srcP in [self.source5, self.source22, self.source6]:
            for tgtP in [self.target6, self.target33]:
                for syn_type in 'excitatory', 'inhibitory':
                    for allow_self_connections in True, False:
                        prj = nest.Projection(srcP, tgtP, nest.AllToAllConnector(allow_self_connections=allow_self_connections), target=syn_type, label="connector")
                        assert len(prj.connection_manager.sources) == len(prj.connection_manager.targets)
                        weights = []
                        for conn in prj.connections:
                            src = conn.source
                            port = conn.port
                            ###print "--------", prj.label, srcP.label, tgtP.label, src, tgt
                            ###print nest.nest.GetConnections([src],'static_synapse') ###
                            weights.append(get_weight(src, port, prj.plasticity_name))
                        assert weights == [0.0]*len(prj.connection_manager)
    
    def testOneToOne(self):
        """For all connections created with "OneToOne" it should be possible to obtain the weight using pyneuron.getWeight()"""
        prj = nest.Projection(self.source33, self.target33, nest.OneToOneConnector())
        self.assertEqual(len(prj), self.source33.size)
        self.assertRaises(common.InvalidDimensionsError, nest.Projection, self.source33, self.target6, nest.OneToOneConnector())
        
    def testDistanceDependentProbability(self):
        """For all connections created with "distanceDependentProbability"..."""
        # Test should be improved..."
        for rng in (None, nest.NumpyRNG(12345), nest.NativeRNG(12345)):
            for expr in ('exp(-d/2)', 'd < 1.1'):
                for allow_self_connections in True, False:
                    for periodic_boundaries in True, False:
                        connector = nest.DistanceDependentProbabilityConnector(d_expression=expr,
                                                                               allow_self_connections=allow_self_connections,
                                                                               periodic_boundaries=periodic_boundaries)
                        for srcP in self.source33, self.target33:
                            prj = nest.Projection(srcP, self.target33,
                                                  connector,
                                                  rng=rng)
                            assert (0 < len(prj) < len(srcP)*len(self.target33)), \
                            "len(prj) = %d, len(srcP)*len(self.target33) = %d, allow_self_connections = %s" % (len(prj), len(srcP)*len(self.target33), allow_self_connections)

    def testFixedNumberPost(self):
        for rng in (None, nest.NumpyRNG(12345)): #, nest.NativeRNG(12345)):
            for srcP in [self.source5, self.source22, self.source6]:
                for tgtP in [self.target6, self.target33]:
                    for asc in True, False:
                        prj = nest.Projection(srcP, tgtP,
                                              nest.FixedNumberPostConnector(n=4, allow_self_connections=asc),
                                              label="connector", rng=rng)
                        assert len(prj.connection_manager.sources) == len(prj.connection_manager.targets), "src=%s, tgt=%s" % (prj._sources, prj._targets)
                        assert prj.getWeights('list') == [0.0, 0.0, 0.0, 0.0]*len(srcP), str(prj.getWeights('list'))
    
    def testFixedNumberPre(self):
        for rng in (None, nest.NumpyRNG(12345)): #, nest.NativeRNG(12345)):
            for srcP in [self.source5, self.source22, self.source6]:    
                for tgtP in [self.target6, self.target33]:
                    for asc in True, False:
                        prj = nest.Projection(srcP, tgtP,
                                              nest.FixedNumberPreConnector(n=4, weights=0.5, allow_self_connections=asc),
                                              label="connector",
                                              rng=rng)
                        assert len(prj.connection_manager.sources) == len(prj.connection_manager.targets), "src=%s, tgt=%s" % (prj._sources, prj._targets)
                        assert prj.getWeights('list') == [0.5, 0.5, 0.5, 0.5]*len(tgtP), str(prj.getWeights('list'))

    def testFixedProbability(self):
        """For all connections created with "fixedProbability" it should be possible to obtain the weight using .getWeight()"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                for rng in None, random.NumpyRNG(), random.NativeRNG():
                    prj = nest.Projection(srcP, tgtP, nest.FixedProbabilityConnector(0.5), rng=rng) 
                    self.assertEqual(len(prj.connection_manager.sources),
                                     len(prj.connection_manager.targets))
                    weights = []
                    for c in prj.connections:
                        #print nest.nest.GetConnections([src],[tgt])
                        weights.append(get_weight(c.source, c.port, prj.plasticity_name))
                    assert weights == [0.]*len(prj.connection_manager)
               
    def testFixedProbabilityWithoutSelfConnections(self):
        for tgtP in [self.target6, self.target33]:
            srcP = tgtP
            c = nest.FixedProbabilityConnector(1.0, allow_self_connections=False)
            prj = nest.Projection(srcP, tgtP, c)
            assert len(prj.connection_manager.sources) == len(prj.connection_manager.targets)
            weights = []
            for c in prj.connections:
                weights.append(get_weight(c.source, c.port, prj.plasticity_name))
            assert weights == [0.]*len(prj.connection_manager)

                    
    def testFromList(self):
        conn_list_22_33 = [([0,0], [0,2], 0.25, 0.5),
                           ([0,0], [1,2], 0.5, 0.5),
                           ([1,0], [2,2], 0.125, 0.1)]
        prj = nest.Projection(self.source22, self.target33, nest.FromListConnector(conn_list_22_33))
        assert prj.getWeights('list') == [0.25, 0.5, 0.125], str(prj.getWeights('list'))
        assert prj.getDelays('list') == [0.5, 0.5, 0.1], str(prj.getDelays('list'))
                    
    def testSaveAndLoad(self):
        prj1 = nest.Projection(self.source22, self.target33, nest.AllToAllConnector())
        prj1.setDelays(0.2)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp")
        prj2 = nest.Projection(self.source22, self.target33, nest.FromFileConnector("connections.tmp"))
        w1 = []; w2 = []; d1 = []; d2 = [];
        # For a connections scheme saved and reloaded, we test if the connections, their weights and their delays
        # are equal.
        for c in prj1.connections:
            src, tgt = c.source,c.port
            w1.append(get_weight(src, tgt, prj1.plasticity_name))
            d1.append(get_delay(src, tgt, prj1.plasticity_name))
        for c in prj1.connections:
            src, tgt = c.source,c.port
            w2.append(get_weight(src, tgt, prj2.plasticity_name))
            d2.append(get_delay(src, tgt, prj2.plasticity_name))
        assert (w1 == w2) and (d1 == d2)

    def testInitialWeightsFromPositiveArray(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                for syn_type in 'excitatory', 'inhibitory':
                    weights_in = numpy.random.uniform(0.1,0.2,len(srcP)*len(tgtP))
                    connector = nest.AllToAllConnector(weights=weights_in)
                    if syn_type == 'excitatory' or "cond" in tgtP.celltype.__class__.__name__:
                        prj = nest.Projection(srcP, tgtP, connector, target=syn_type)
                        weights_out = numpy.array(prj.getWeights(format='list'))
                        #if syn_type == 'inhibitory':
                        #    weights_out *= -1
                        assert arrays_almost_equal(weights_in, weights_out, 1e-9), '(%s) %s != %s' % (syn_type, weights_in, weights_out)
                    else:
                        self.assertRaises(common.InvalidWeightError, nest.Projection, srcP, tgtP, connector, target=syn_type)

    def testInitialInhibitoryWeightsFromNegativeArray(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                weights_in = numpy.random.uniform(-0.1,-0.2,len(srcP)*len(tgtP))
                connector = nest.AllToAllConnector(weights=weights_in)
                if "curr" in tgtP.celltype.__class__.__name__:
                    prj = nest.Projection(srcP, tgtP, connector, target='inhibitory')
                    weights_out = numpy.array(prj.getWeights(format='list'))
                    assert arrays_almost_equal(weights_in, weights_out, 1e-9), '%s != %s' % (weights_in, weights_out)
                else:
                    self.assertRaises(common.InvalidWeightError, nest.Projection, srcP, tgtP, connector, target='inhibitory')
                
    def testInitialExcitatoryWeightsFromNegativeArray(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                weights_in = numpy.random.uniform(-0.1,-0.2,len(srcP)*len(tgtP))
                connector = nest.AllToAllConnector(weights=weights_in)
                self.assertRaises(common.InvalidWeightError, nest.Projection, srcP, tgtP, connector, target='excitatory')
                
    def testInitialWeightsFromMixedArray(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                weights_in = numpy.random.uniform(-0.1,0.1,len(srcP)*len(tgtP))
                connector = nest.AllToAllConnector(weights=weights_in)
                self.assertRaises(common.InvalidWeightError, nest.Projection, srcP, tgtP, connector)
    
    def testInitialNegativeWeight(self):
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                connector = nest.AllToAllConnector(weights=-1.23)
                self.assertRaises(common.InvalidWeightError, nest.Projection, srcP, tgtP, connector, target='excitatory')
            prj = nest.Projection(srcP, self.target33, connector, target='inhibitory') # current
            assert arrays_almost_equal(numpy.array(prj.getWeights(format='list')), -1.23*numpy.ones(len(srcP)*len(tgtP)), 1e-9)
            self.assertRaises(common.InvalidWeightError, nest.Projection, srcP, self.target6, connector, target='inhibitory') # conductance
            
                
    def testInitialDistanceDependentWeights(self):
        connector = nest.DistanceDependentProbabilityConnector("d<1.2", weights="exp(-d)", delays="0.1 + 0.1*d")
        prj = nest.Projection(self.source33, self.target33, connector)
        for c in prj.connections:
            src_id, tgt_id, port = c.source, c.target, c.port
            src_addr = prj.pre.locate(src_id)
            tgt_addr = prj.post.locate(tgt_id)
            nest_connection = nest.nest.GetConnection([src_id], prj.plasticity_name, port)
            #print src_addr, tgt_addr, nest_connection
            distance = numpy.sqrt( (src_addr[0]-tgt_addr[0])**2 + (src_addr[1]-tgt_addr[1])**2 )
            self.assertEqual(0.001*nest_connection['weight'], numpy.exp(-distance))
            self.assertEqual(nest_connection['delay'], 0.1 + 0.1*distance)
                
class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), randomizeWeights() and
    randomizeDelays() methods of the Projection class."""

    def setUp(self):
        nest.setup(min_delay=0.1)
        nest.Population.nPop = 0
        self.target33 = nest.Population((3,3),nest.IF_curr_alpha)
        self.target6  = nest.Population((6,),nest.IF_curr_alpha)
        self.source5  = nest.Population((5,),nest.SpikeSourcePoisson)
        self.source22 = nest.Population((2,2),nest.SpikeSourcePoisson)
        self.prjlist = []
        self.distrib_Numpy = random.RandomDistribution(rng=random.NumpyRNG(12345),distribution='uniform',parameters=(0.1,0.5)) 
        for tgtP in [self.target6, self.target33]:
            for srcP in [self.source5, self.source22]:
                for method in (nest.AllToAllConnector(), nest.FixedProbabilityConnector(p_connect=0.5)):
                    self.prjlist.append(nest.Projection(srcP,tgtP,method) )

    def testSetWeightsToSingleValue(self):
        """Weights set using setWeights() should be retrievable with .getWeight()"""
        for prj in self.prjlist:
            prj.setWeights(1.234)
            for c in prj.connections:
                src, tgt = c.source, c.port
                assert get_weight(src, tgt, prj.plasticity_name) == 1234.0 # note the difference in units between pyNN and NEST

    #def testSetAndGetID(self):
        # Small test to see if the ID class is working
        #self.target33[0,2].set({'tau_m' : 15.1})
        #assert (self.target33[0,2].get('tau_m') == 15.1)
    
    def testSetWeightsWithList(self):
        for prj in self.prjlist:
            weights_in = self.distrib_Numpy.next(len(prj))
            prj.setWeights(weights_in)
            weights_out = []
            for c in prj.connections:
                src, tgt = c.source, c.port
                weights_out.append(0.001*get_weight(src, tgt, prj.plasticity_name)) # units conversion
            self.assert_(arrays_almost_equal(weights_in, weights_out, 1e-8), "%s != %s" % (weights_in, weights_out))
            
    def testRandomizeWeights(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = nest.Projection(self.source5, self.target33, nest.AllToAllConnector())
        prj1.randomizeWeights(self.distrib_Numpy)
        w1 = []; w2 = [];
        for c in prj1.connections:
            src, tgt = c.source, c.port
            w1.append(get_weight(src, tgt, prj1.plasticity_name))
        prj1.randomizeWeights(self.distrib_Numpy)        
        for c in prj1.connections:
            src, tgt = c.source, c.port
            w2.append(get_weight(src, tgt, prj1.plasticity_name))
        self.assertNotEqual(w1,w2)
        
    def testRandomizeDelays(self):
        # The probability of having two consecutive weights vector that are equal should be 0
        prj1 = nest.Projection(self.source5, self.target33, nest.AllToAllConnector())
        prj1.randomizeDelays(self.distrib_Numpy)
        d1 = []; d2 = [];
        for c in prj1.connections:
            src, tgt = c.source, c.port
            d1.append(get_delay(src,tgt, prj1.plasticity_name))
        prj1.randomizeDelays(self.distrib_Numpy)        
        for c in prj1.connections:
            src, tgt = c.source, c.port
            d2.append(get_delay(src,tgt, prj1.plasticity_name))
        self.assertNotEqual(d1,d2)

class ProjectionGetTest(unittest.TestCase):
    """Tests of the getWeights(), getDelays() methods of the Projection class."""

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
                for method in (nest.AllToAllConnector(), nest.FixedProbabilityConnector(p_connect=0.5)):
                    self.prjlist.append(nest.Projection(srcP,tgtP,method))

    def testGetWeightsWithList(self):
        for prj in self.prjlist:
            weights_in = self.distrib_Numpy.next(len(prj))
            prj.setWeights(weights_in)
            weights_out = numpy.array(prj.getWeights(format='list'))
            self.assert_(arrays_almost_equal(weights_in, weights_out, 1e-8), "%s != %s" % (weights_in, weights_out))
            
    def testGetWeightsWithArray(self):
        """Making 1D and removing weights <= 0 should turn the array format of getWeights()
        into the list format."""
        for prj in self.prjlist:
            weights_in = self.distrib_Numpy.next(len(prj))
            prj.setWeights(weights_in)
            weights_out = numpy.array(prj.getWeights(format='array')).flatten()
            weights_out = weights_out.compress(weights_out>0)
            self.assert_(arrays_almost_equal(weights_in, weights_out, 1e-8), "%s != %s" % (weights_in, weights_out))
            
            
class IDTest(unittest.TestCase):
    """Tests of the ID class."""
    
    def setUp(self):
        nest.setup(max_delay=0.5)
        nest.Population.nPop = 0
        self.pop1 = nest.Population((5,),nest.IF_curr_alpha,{'tau_m':10.0})
        self.pop2 = nest.Population((5,4),nest.IF_curr_exp,{'v_reset':-60.0})
        self.pop3 = nest.Population((10,), nest.IF_cond_alpha)
        self.pop4 = nest.Population((3,4,5), 'iaf_neuron')
    
    def testIDSetAndGet(self):
        self.pop1[3].tau_m = 20.0
        self.pop2[3,2].v_reset = -70.0
        self.pop3[5].tau_m = -55.0
        ifcell_params = nest.nest.GetStatus([self.pop1[3]])[0]
        self.assertEqual(20.0, ifcell_params['tau_m'])
        self.assertEqual(20.0, self.pop1[3].tau_m)
        self.assertEqual(10.0, self.pop1[0].tau_m)
        self.assertEqual(-70.0, self.pop2[3,2].v_reset)
        self.assertEqual(-60.0, self.pop2[0,0].v_reset)
        self.assertAlmostEqual(-55.0, self.pop3[5].tau_m, 9)

    def testGetCellClass(self):
        self.assertEqual(nest.IF_curr_alpha, self.pop1[0].cellclass)
        self.assertEqual(nest.IF_curr_exp, self.pop2[4,3].cellclass)
        self.assertEqual('iaf_neuron', self.pop4[0,0,0].cellclass)
        
    def testSetAndGetPosition(self):
        self.assert_((self.pop2[0,2].position == (0.0,2.0,0.0)).all())
        new_pos = (0.5,1.5,0.0)
        self.pop2[0,2].position = new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())
        new_pos = (-0.6,3.5,-100.0) # check that position is set-by-value from new_pos
        self.assert_((self.pop2[0,2].position == (0.5,1.5,0.0)).all())

class SynapseDynamicsTest(unittest.TestCase):
    
    def setUp(self):
        nest.setup(max_delay=0.5)
        nest.Population.nPop = 0
        self.target33 = nest.Population((3,3),nest.IF_curr_alpha, label="target33")
        self.target6  = nest.Population((6,),nest.IF_cond_exp, label="target6")
        self.source5  = nest.Population((5,),nest.SpikeSourcePoisson, label="source5")
        self.source6  = self.target6
        self.source22 = nest.Population((2,2),nest.SpikeSourcePoisson, label="source22")
        self.source33 = nest.Population((3,3),nest.SpikeSourcePoisson, label="source33")
        self.connectors = [nest.AllToAllConnector(weights=0.1, delays=0.2),
                           nest.FixedProbabilityConnector(1.0, weights=0.1, delays=0.2)]

    def testCreateProjectionWithTsodyksMarkramSynapses(self):
        sd = nest.SynapseDynamics(fast=nest.TsodyksMarkramMechanism())
        for srcP in [self.source5, self.source22, self.source6]:
            for tgtP in [self.target6, self.target33]:
                for conn  in self.connectors:
                    prj = nest.Projection(srcP, tgtP,
                                          conn,
                                          label="connector",
                                          synapse_dynamics=sd)
                    assert prj.getWeights('list') == [0.1]*(len(srcP)*len(tgtP)), "%s != %s" % (prj.getWeights('list'),[0.1]*(len(srcP)*len(tgtP)))
                    assert prj.getDelays('list') == [0.2]*(len(srcP)*len(tgtP))
                    assert prj.getSynapseDynamics('U', 'list', gather=False) == [0.5]*(len(srcP)*len(tgtP))

    def testCreateSimpleSTDPConnection(self):
        for wd in [nest.AdditiveWeightDependence(w_max=0.7654),
                   nest.GutigWeightDependence(w_max=0.7654)]:
            stdp = nest.STDPMechanism(timing_dependence=nest.SpikePairRule(),
                                      weight_dependence=wd)
            sd = nest.SynapseDynamics(slow=stdp)
            for srcP in [self.source5, self.source22, self.source6]:
                for tgtP in [self.target6, self.target33]:
                    for conn  in self.connectors:
                        prj = nest.Projection(srcP, tgtP,
                                              conn,
                                              label="connector",
                                              synapse_dynamics=sd)
                        assert prj.getWeights('list') == [0.1]*(len(srcP)*len(tgtP)), "%s != %s" % (prj.getWeights('list'),[0.1]*(len(srcP)*len(tgtP)))
                        assert prj.getDelays('list') == [0.2]*(len(srcP)*len(tgtP))
                        assert prj.getSynapseDynamics('Kplus', 'list', gather=False) == [0.0]*(len(srcP)*len(tgtP))
                        assert nest.nest.GetDefaults(prj.plasticity_name)['Wmax'] == 0.7654*1000.0

    def test_nonzero_wmin(self):
        self.assertRaises(Exception, nest.AdditiveWeightDependence, w_min=0.01)
        self.assertRaises(Exception, nest.MultiplicativeWeightDependence, w_min=0.01)
        self.assertRaises(Exception, nest.AdditivePotentiationMultiplicativeDepression, w_min=0.01)
        self.assertRaises(Exception, nest.GutigWeightDependence, w_min=0.01)

class ElectrodesTest(unittest.TestCase):
    
    def test_NoisyCurrentSource(self):
        # just check no Exceptions are raised, for now.
        source = nest.NoisyCurrentSource(0.0, 1.0, start=0.0, stop=100.0, frozen=False)
        cells = nest.create(nest.IF_curr_exp, {}, 5)
        source.inject_into(cells)
        for cell in cells:
            cell.inject(source)

class ConnectionClassTest(unittest.TestCase):
    
    def setUp(self):
        nest.setup(timestep=0.1, min_delay=0.1)
        self.cell1 = nest.create(nest.IF_curr_exp, {})
        self.cell2 = nest.create(nest.IF_cond_alpha, {})
    
    def test_get_set_weight(self):
        conn = nest.connect(self.cell1, self.cell2)[0]
        self.assertEqual(conn.weight, 0.0)
        conn.weight = 0.123
        self.assertEqual(conn.weight, 0.123)
        self.assertEqual(nest.nest.GetConnection([conn.source], conn.parent.synapse_model, conn.port)['weight'], 123.0) 

    def test_get_set_delay(self):
        conn = nest.connect(self.cell1, self.cell2)[0]
        self.assertEqual(conn.delay, nest.get_min_delay())
        conn.delay = 2*nest.get_time_step()
        self.assertEqual(conn.delay, 2*nest.get_time_step())
        conn.delay = 1.6*nest.get_time_step()
        self.assertEqual(conn.delay, 2*nest.get_time_step())
        conn.delay = 1.4*nest.get_time_step()
        self.assertEqual(conn.delay, 2*nest.get_time_step())
        conn.delay = 1.4*nest.get_time_step()
        self.assertRaises(nest.nest.NESTError, setattr, conn, 'delay',  0.9*nest.get_time_step())

class RecorderTest(unittest.TestCase):
    
    def setUp(self):
        nest.setup()
    
    def test_create_spike_recorder_in_memory(self):
        rec = nest.simulator.Recorder('spikes', file=False)
        rec._create_device()
        params = nest.nest.GetStatus([rec._device])[0]
        self.assertEqual(params['model'], 'spike_detector')
        self.assertEqual(params['to_file'], False)
        self.assertEqual(params['to_memory'], True)
        
    def test_premature_get(self):
        rec = nest.simulator.Recorder('spikes', file=False)
        self.assertRaises(Exception, rec.get)
        
    def test_get_before_run(self):
        rec1 = nest.simulator.Recorder('spikes', file=False)
        rec2 = nest.simulator.Recorder('spikes', file=None) # creates temporary file
        cell = nest.create(nest.IF_curr_exp, {})
        for rec in rec1, rec2:
            rec.record([cell])
            data = rec.get()
            self.assertEqual(data.shape, (0,2))

    def test_merge_files(self):
        rec = nest.simulator.Recorder('spikes', file=None)
        cells = nest.create(nest.IF_curr_exp, {}, n=5)
        rec.record(cells)
        nest.run(1.0)
        filename = nest.simulator._merge_files(rec._device, False)

    def test_write_not_compatible(self):
        rec1 = nest.simulator.Recorder('spikes', file=False)
        rec2 = nest.simulator.Recorder('spikes', file='nest_recorder_test.tmp')
        cell = nest.create(nest.IF_curr_exp, {})
        for rec in rec1, rec2:
            rec.record([cell])
        nest.run(1.0)
        rec2.write(compatible_output=False)
        self.assertRaises(Exception, rec1.write, compatible_output=False) # writing for in-memory data not yet supported
        os.remove('nest_recorder_test.tmp')
        
    def test_write_invalid_file(self):
        rec = nest.simulator.Recorder('spikes', file=None)
        cell = nest.create(nest.IF_curr_exp, {})
        rec.record([cell])
        nest.run(1.0)
        self.assertRaises(Exception, rec.write, file={}, compatible_output=False)

    def test_manipulate_header(self):
        rec = nest.simulator.Recorder('spikes', file='nest_recorder_test.tmp')
        cell = nest.create(nest.IF_curr_exp, {})
        rec.record([cell])
        nest.run(1.0)
        rec.write(compatible_output=True)
        header = rec._get_header('nest_recorder_test.tmp')
        assert isinstance(header, dict)
        self.assertEqual(float(header['dt']), 0.1)
        
        rec._strip_header('nest_recorder_test.tmp', 'nest_recorder_test1.tmp')
        header = rec._get_header('nest_recorder_test1.tmp')
        self.assertEqual(len(header), 0)
        
        os.remove('nest_recorder_test.tmp')
        os.remove('nest_recorder_test1.tmp')
        
        rec._get_header('awiulclwiufhdp3948tdw3.dat') # does not exist

if __name__ == "__main__":
    unittest.main()
    
