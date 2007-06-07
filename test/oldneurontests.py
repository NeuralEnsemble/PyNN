"""
Unit tests for pyNN/oldneuron.py.
$Id$
"""

import pyNN.oldneuron as neuron
import pyNN.common as common
import pyNN.random as random
from pyNN.oldneuron import HocToPy
import unittest, sys, numpy

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def tearDown(self):
        for i in range(neuron.hoc_cells):
            cmd = 'objref cell%d\n' % i
            neuron.logfile.write(cmd)
            neuron.hoc.execute(cmd)
        neuron.hoc_cells = 0
    
    def testCreateStandardCell(self):
        """create(): First cell created should be called 'cell0'."""
        ifcell = neuron.create(neuron.IF_curr_alpha)
        assert ifcell == 'cell0', 'Failed to create standard cell'
        
    def testCreateStandardCells(self):
        """create(): Creating multiple cells should return a list of objref names"""
        ifcells = neuron.create(neuron.IF_curr_alpha,n=10)
        assert ifcells == ['cell0', 'cell1', 'cell2', 'cell3', 'cell4', 'cell5', 'cell6', 'cell7', 'cell8', 'cell9'], 'Failed to create 10 standard cells'
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        self.assertRaises(AssertionError, neuron.create, neuron.IF_curr_alpha, n=-1)
       
    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with HocToPy.get()"""
        ifcell = neuron.create(neuron.IF_curr_alpha,{'tau_syn_E':3.141592654})
        self.assertAlmostEqual(HocToPy.get('%s.esyn.tau' % ifcell, 'float'), 3.141592654, places=5)
    
    def testCreateNEURONCell(self):
        """create(): First cell created should be called 'cell0'."""
        ifcell = neuron.create('StandardIF',{'syn_type':'current','syn_shape':'exp'})
        assert ifcell == 'cell0', 'Failed to create NEURON-specific cell'
    
#    def testCreateNonStandardCell(self):
#        """create(): Trying to create a cell type which is not a method of StandardCells should raise an AttributeError."""
#        self.assertRaises(AttributeError, neuron.create, 'qwerty')
    
    #def testCreateWithInvalidParameter(self):
    #    """create(): Creating a cell with an invalid parameter should raise an Exception."""
    #    self.assertRaises(common.InvalidParameterError, neuron.create, neuron.IF_curr_alpha, {'tau_foo':3.141592654})        


# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        self.postcells = neuron.create(neuron.IF_curr_alpha,n=3)
        self.precells = neuron.create(neuron.SpikeSourcePoisson,n=5)
        
    def tearDown(self):
        hoc_commands = []
        for i in range(neuron.hoc_cells):
            hoc_commands.append('objref cell%d\n' % i)
        for i in range(neuron.hoc_netcons):
            hoc_commands.append('objref nc%d\n' % i)
        for cmd in hoc_commands:
            neuron.logfile.write(cmd)
            neuron.hoc.execute(cmd)
        neuron.hoc_cells = 0
        neuron.hoc_netcons = 0
        
    def testConnectTwoCells(self):
        """connect(): The first connection created should have id 0."""
        conn = neuron.connect(self.precells[0],self.postcells[0])
        assert conn == [0], 'Error creating connection'
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        conn_id = neuron.connect(self.precells[0],self.postcells[0],weight=0.1234)
        weight = HocToPy.get('nc%d.weight' % conn_id[0], 'float')
        assert weight == 0.1234, "Weight set (0.1234) does not match weight retrieved (%s)" % weight
    
    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        conn_id = neuron.connect(self.precells[0],self.postcells[0],delay=4.321)
        delay = HocToPy.get('nc%d.delay' % conn_id[0], 'float')
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
        self.assertRaises(common.ConnectionError, neuron.connect, 'cell12345', self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, neuron.connect, self.precells[0], 'cell45678')
    
    def testInvalidSourceId(self):
        """connect(): sources must be strings."""
        self.precells.append(74367598)
        self.assertRaises(AssertionError, neuron.connect, self.precells, self.postcells)
    
    def testInvalidTargetId(self):
        """connect(): targets must be strings."""
        self.postcells.append([])
        self.assertRaises(AssertionError, neuron.connect, self.precells, self.postcells)

# ==============================================================================
class SetValueTest(unittest.TestCase):
    
    def setUp(self):
        self.cells = neuron.create(neuron.IF_curr_exp,n=10)
        
    def testSetFloat(self):
        neuron.set(self.cells,neuron.IF_curr_exp,'tau_m',35.7)
        for cell in self.cells:
            assert HocToPy.get('%s.tau_m' % cell, 'float') == 35.7
            
    def testSetString(self):
        neuron.set(self.cells,neuron.IF_curr_exp,'syn_shape','alpha')
        # note you can't actually change the synaptic shape this way
        for cell in self.cells:
            assert HocToPy.get('%s.syn_shape' % cell, 'string') == 'alpha'

    def testSetDict(self):
        neuron.set(self.cells,neuron.IF_curr_exp,{'tau_m':35.7,'syn_shape':'qwerty'})
        for cell in self.cells:
            assert HocToPy.get('%s.syn_shape' % cell, 'string') == 'qwerty'
            assert HocToPy.get('%s.tau_m' % cell, 'float') == 35.7

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
        """Population.__init__(): should set the NetLayer.label attribute in hoc."""
        net = neuron.Population((3,3),neuron.IF_curr_alpha)
        # shouldn't really have three assertions in one test but I'm lazy
        assert net.label == HocToPy.get('%s.label' % net.label, 'string')
        assert HocToPy.get('%s.dimensions()' % net.label, 'integer') == 2
        assert HocToPy.get('%s.size()' % net.label, 'integer') == 3
           
    def testInitWithParams(self):
        """Population.__init__(): Parameters set on creation should be the same as retrieved with HocToPy.get()"""
        net = neuron.Population((3,3),neuron.IF_curr_alpha,{'tau_syn_E':3.141592654})
        tau_syn = HocToPy.get('%s.cell[0][0].esyn.tau' % net.label)
        self.assertAlmostEqual(tau_syn, 3.141592654, places=5)
    
    def testInitWithLabel(self):
        """Population.__init__(): A label set on initialisation should be retrievable with the Population.label attribute."""
        net = neuron.Population((3,3),neuron.IF_curr_alpha,label='iurghiushrg')
        assert net.label == 'iurghiushrg'
    
#    def testInvalidCellType(self):
#        """Population.__init__(): Trying to create a cell type which is not a method of StandardCells should raise an AttributeError."""
#        self.assertRaises(AttributeError, neuron.Population, (3,3), 'qwerty')
        
    def testNonSquareDimensions(self):
        """Population.__init__(): At present all dimensions must be the same size."""
        self.assertRaises(common.InvalidDimensionsError, neuron.Population, (3,2), neuron.IF_curr_alpha)

    def testInitWithNonStandardModel(self):
        """Population.__init__(): should set the NetLayer.label attribute in hoc."""
        net = neuron.Population((3,3),'StandardIF',{'syn_type':'current','syn_shape':'exp'})
        assert net.label == HocToPy.get('%s.label' % net.label, 'string')
        assert HocToPy.get('%s.dimensions()' % net.label, 'integer') == 2
        assert HocToPy.get('%s.size()' % net.label, 'integer') == 3

# ==============================================================================
class PopulationSetTest(unittest.TestCase):
        
    def setUp(self):
        neuron.Population.nPop = 0
        self.net = neuron.Population((3,3),neuron.IF_curr_alpha)
        self.net2 = neuron.Population((5,),'StandardIF',{'syn_type':'current','syn_shape':'exp'})
    
    def testSetFromDict(self):
        """Population.set(): Parameters set in a dict should all be retrievable using HocToPy.get()"""
        self.net.set({'tau_m':43.21})
        assert HocToPy.get('%s.cell[2][1].tau_m' % self.net.label, 'float') == 43.21
    
    def testSetFromPair(self):
        """Population.set(): A parameter set as a string,value pair should be retrievable using HocToPy.get()"""
        self.net.set('tau_m',12.34)
        assert HocToPy.get('%s.cell[2][1].tau_m' % self.net.label, 'float') == 12.34
    
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
        assert HocToPy.get('%s.cell[2].tau_m' % self.net2.label, 'float') == 43.21
        
    def testTSet(self):
        """Population.tset(): The valueArray passed should be retrievable using HocToPy.get() on all nodes."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.net.tset('i_offset', array_in)
        array_out = numpy.zeros((3,3),float)
        for i in 0,1,2:
            for j in 0,1,2:
                array_out[i,j]= HocToPy.get('%s.cell[%d][%d].stim.amp' % (self.net.label,i,j),'float')
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
        self.assertNotEqual(HocToPy.get('%s.cell[2][1].tau_m' % self.net.label),
                            HocToPy.get('%s.cell[1][2].tau_m' % self.net.label))
        
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
                output_values[i,j] = HocToPy.get('%s.cell[%d][%d].cell.cm' % (self.net.label,i,j),'float')
        input_values = rd2.next(9)
        output_values = output_values.reshape((9,))
        for i in range(9):
            self.assertAlmostEqual(input_values[i],output_values[i],places=5)
        
# ==============================================================================
class PopulationCallTest(unittest.TestCase): # to write later
    """Tests of the _call() and _tcall() methods of the Population class."""
    pass

# ==============================================================================
class PopulationRecordTest(unittest.TestCase): # to write later
    """Tests of the record(), record_v(), printSpikes(), print_v() and
       meanSpikeCount() methods of the Population class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.net = neuron.Population((3,3),neuron.IF_curr_alpha)
        
    def testRecordAll(self):
        """Population.record(): not a full test, just checking there are no Exceptions raised."""
        self.net.record()
        
    def testRecordInt(self):
        """Population.record(n): not a full test, just checking there are no Exceptions raised."""
        self.net.record(5)
        
    def testRecordWithRNG(self):
        """Population.record(n,rng): not a full test, just checking there are no Exceptions raised."""
        self.net.record(5,random.NumpyRNG())
        
    def testRecordList(self):
        """Population.record(list): not a full test, just checking there are no Exceptions raised."""
        self.net.record([self.net[(2,2)],self.net[(1,2)],self.net[(0,0)]])

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
        self.target33 = neuron.Population((3,3),neuron.IF_curr_alpha)
        self.target6  = neuron.Population((6,),neuron.IF_curr_alpha)
        self.source5  = neuron.Population((5,),neuron.SpikeSourcePoisson)
        self.source22 = neuron.Population((2,2),neuron.SpikeSourcePoisson)

    def testAllToAll(self):
        """For all connections created with "allToAll" it should be possible to obtain the weight using pyneuron.getWeight()"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, "allToAll")
                prj1.setWeights(1.234)
                weights = []
                for connection_id in prj1.connections():
                    weights.append(HocToPy.get('%s.nc%s.weight' % (prj1.label,connection_id), 'float'))
                assert weights == [1.234]*len(prj1)
        
    def testFixedProbability(self):
        """For all connections created with "fixedProbability" it should be possible to obtain the weight using pyneuron.getWeight()"""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = neuron.Projection(srcP, tgtP, "fixedProbability", 0.9)
                weights = []
                for connection_id in prj1.connections():
                    if HocToPy.bool('object_id(%s.nc%s)' % (prj1.label,connection_id)):
                        weights.append(HocToPy.get('%s.nc%s.weight' % (prj1.label,connection_id),'float'))
                assert 0 < len(weights) < len(prj1)
 

class ProjectionConnectionTest(unittest.TestCase):
    """Tests of the connection attribute and connections() method of the Projection class."""
    
    def setUp(self):
        neuron.Population.nPop = 0
        self.pop1 = neuron.Population((5,),neuron.IF_curr_alpha)
        self.pop2 = neuron.Population((4,4),neuron.IF_curr_alpha)    
        self.pop3 = neuron.Population((3,3,3),neuron.IF_curr_alpha)
        self.prj23 = neuron.Projection(self.pop2,self.pop3,"allToAll")
        self.prj11 = neuron.Projection(self.pop1,self.pop1,"fixedProbability",0.5)
        
    def testFullAddress(self):
        assert self.prj23.connection[(3,1),(2,0,1)] == "[3][1][2][0][1]"
        assert self.prj23.connection[(3,3),(2,2,2)] == "[3][3][2][2][2]"
        
    def testPreIDPostID(self):
        assert self.prj23.connection[0,0] == "[0][0][0][0][0]"
        assert self.prj23.connection[0,26] == "[0][0][2][2][2]"
        assert self.prj23.connection[0,25] == "[0][0][2][2][1]"
        assert self.prj23.connection[15,0] == "[3][3][0][0][0]"
        assert self.prj23.connection[14,0] == "[3][2][0][0][0]"
        assert self.prj23.connection[13,19] == "[3][1][2][0][1]"
        
    def testSingleID(self):
        assert self.prj23.connection[0] == "[0][0][0][0][0]"
        assert self.prj23.connection[26] == "[0][0][2][2][2]"
        assert self.prj23.connection[25] == "[0][0][2][2][1]"
        assert self.prj23.connection[27] == "[0][1][0][0][0]"
        assert self.prj23.connection[53] == "[0][1][2][2][2]"
        assert self.prj23.connection[52] == "[0][1][2][2][1]"
        assert self.prj23.connection[431] == "[3][3][2][2][2]"
        assert self.prj23.connection[377] == "[3][1][2][2][2]"
        assert self.prj23.connection[370] == "[3][1][2][0][1]"
        
        assert self.prj11.connection[0] == "[0][0]"
    
class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), setThreshold(),
       randomizeWeights() and randomizeDelays() methods of the Projection class."""

    pass

if __name__ == "__main__":
    sys.argv = ['./nrnpython']
    neuron.setup()
    unittest.main()
    neuron.end()
    
