"""
Unit tests for all simulators
$Id:$
"""

import sys
import unittest
import numpy
import os
from pyNN import common, random

def assert_arrays_almost_equal(a, b, threshold):
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        raise unittest.TestCase.failureException(err_msg)

def is_local(cell):
    return True # implement this for testing distributed simulation

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def testCreateStandardCell(self):
        for cellclass in sim.list_standard_models():
            ifcell = sim.create(cellclass)
            assert isinstance(ifcell, common.IDMixin)
        
    def testCreateStandardCells(self):
        for cellclass in sim.list_standard_models():
            ifcells = sim.create(cellclass, n=10)
            self.assertEqual(len(ifcells), 10)
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        self.assertRaises(AssertionError, sim.create, sim.IF_curr_alpha, n=-1)

    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with the top-level HocObject"""
        ifcell = sim.create(sim.IF_curr_alpha,{'tau_syn_E':3.141592654})
        self.assertAlmostEqual(ifcell.tau_syn_E, 3.141592654, places=9)
        
    def testCreateInvalidCell(self):
        """create(): Trying to create a cell type which is not a standard cell or
        valid native cell should raise an Exception."""
        self.assertRaises(common.InvalidModelError, sim.create, 'qwerty', n=10)
    
    def testCreateWithInvalidParameter(self):
        """create(): Creating a cell with an invalid parameter should raise an Exception."""
        self.assertRaises(common.NonExistentParameterError, sim.create, sim.IF_curr_alpha, {'tau_foo':3.141592654})  

# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        self.postcells = sim.create(sim.IF_curr_alpha, n=3)
        self.precells = sim.create(sim.SpikeSourcePoisson, n=5)
        
    def tearDown(self):
        pass
        
    def testConnectTwoCells(self):
        conn_list = sim.connect(self.precells[0], self.postcells[0])
        # conn will be an empty list if it does not exist on that node
        self.assertEqual(len(conn_list), 1)
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        conn_list = sim.connect(self.precells[0], self.postcells[0], weight=0.1234)
        if conn_list:
            weight = conn_list[0].weight
            self.assertEqual(weight, 0.1234)
            
    def testConnectTwoCellsWithDelay(self):
        conn_list = sim.connect(self.precells[0], self.postcells[0], delay=4.321)
        if conn_list:
            delay = conn_list[0].delay
            if simulator == 'nest2':
                self.assertEqual(round(delay, 1), 4.4) # NEST rounds delays to the timestep, 0.1 here
            else:
                self.assertEqual(delay, 4.321) 
    
    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n,
        each element being the id number of a netcon."""
        conn_list = sim.connect(self.precells, self.postcells[0])
        # connections are only created on the node containing the post-syn
        self.assertEqual(len(conn_list), len(self.precells))
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        conn_list = sim.connect(self.precells[0], self.postcells)
        cells_on_this_node = len([i for i in self.postcells if is_local(i)])
        self.assertEqual(len(conn_list),  cells_on_this_node)
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        conn_list = sim.connect(self.precells, self.postcells)
        cells_on_this_node = len([i for i in self.postcells if is_local(i)])
        expected_length = cells_on_this_node*len(self.precells)
        self.assertEqual(len(conn_list), expected_length, "%d != %d" % (len(conn_list), expected_length))
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        conn_list = sim.connect(self.precells, self.postcells, p=0.5)
        cells_on_this_node = len([i for i in self.postcells if is_local(i)])
        assert 0 < len(conn_list) < len(self.precells)*cells_on_this_node, 'Number of connections is %d: this is very unlikely (although possible).' % len(conn_list)
    
    def testConnectNonExistentPreCell(self):
        """connect(): Connecting from non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, sim.connect, 12345, self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, sim.connect, self.precells[0], 'cell45678')
    
    def testInvalidSourceId(self):
        """connect(): sources must be integers."""
        self.precells.append('74367598')
        self.assertRaises(common.ConnectionError, sim.connect, self.precells, self.postcells)
    
    def testInvalidTargetId(self):
        """connect(): targets must be integers."""
        self.postcells.append('99.9')
        self.assertRaises(common.ConnectionError, sim.connect, self.precells, self.postcells)
    
    def testConnectTooSmallDelay(self):
        self.assertRaises(common.ConnectionError, sim.connect, self.precells[0], self.postcells[0], delay=1e-12)

# ==============================================================================
class IDSetGetTest(unittest.TestCase):
    """Tests of the ID.__setattr__()`, `ID.__getattr()` `ID.setParameters()`
    and `ID.getParameters()` methods for all available standard cell types
    and for both lone and in-population IDs."""
    
    model_list = []
    default_dp = 5
    decimal_places = {'duration': 2, 'start': 2}
        
    def setUp(self):
        sim.setup()
        self.cells = {}
        self.populations = {}
        if not IDSetGetTest.model_list:
            IDSetGetTest.model_list = sim.list_standard_models()
        for cell_class in IDSetGetTest.model_list:
            self.cells[cell_class.__name__] = sim.create(cell_class, n=2)
            self.populations[cell_class.__name__] = sim.Population(2, cell_class)
    
    def tearDown(self):
        pass
    
    def testSetGet(self):
        """__setattr__(), __getattr__(): sanity check"""
        for cell_class in IDSetGetTest.model_list:
            cell_list = (self.cells[cell_class.__name__][0],
                         self.populations[cell_class.__name__][0])
            parameter_names = cell_class.default_parameters.keys()
            for cell in cell_list:
                for name in parameter_names:
                    if name == 'spike_times':
                        i = [1.0, 2.0]
                        cell.__setattr__(name, i)
                        o = list(cell.__getattr__(name))
                        self.assertEqual(i, o)
                    else:
                        if name == 'v_thresh':
                            if 'v_spike' in parameter_names:
                                i = (cell.__getattr__('v_spike') + max(cell.__getattr__('v_reset'), cell.__getattr__('v_init')))/2
                            elif 'v_init' in parameter_names:
                                i = max(cell.__getattr__('v_reset'), cell.__getattr__('v_init')) + numpy.random.uniform(0.1, 100)
                            else:
                                i = cell.__getattr__('v_reset') + numpy.random.uniform(0.1, 100)
                        elif name == 'v_reset' or name == 'v_init': # v_reset must be less than v_thresh
                            if hasattr(cell, 'v_thresh'):
                                i = cell.__getattr__('v_thresh') - numpy.random.uniform(0.1, 100)
                            else:
                                i = numpy.random.uniform(0.1, 100)
                        elif name == 'v_spike': # v_spike must be greater than v_thresh
                            i = cell.__getattr__('v_thresh') + numpy.random.uniform(0.1, 100)
                        else:
                            i = numpy.random.uniform(0.1, 100) # tau_refrac is always at least dt (=0.1)
                        try:
                            cell.__setattr__(name, i)
                        except Exception, e:
                            raise Exception("%s. %s=%g in %s with %s" % (e, name, i, cell_class, cell.get_parameters()))
                        o = cell.__getattr__(name)
                        self.assertEqual(type(i), type(o), "%s: input: %s, output: %s" % (name, type(i), type(o)))
                        self.assertAlmostEqual(i, o,
                                               IDSetGetTest.decimal_places.get(name, IDSetGetTest.default_dp),
                                               "%s in %s: %s != %s" % (name, cell_class.__name__, i,o))
    
    def testSetGetParameters(self):
        """setParameters(), getParameters(): sanity check"""
        # need to add similar test for native models in the sim-specific test files
        default_dp = 6
        decimal_places = {'duration': 2, 'start': 2}
        for cell_class in IDSetGetTest.model_list:
            cell_list = (self.cells[cell_class.__name__][0],
                         self.populations[cell_class.__name__][0])
            parameter_names = cell_class.default_parameters.keys()
            if 'v_thresh' in parameter_names: # make sure 'v_thresh' comes first
                parameter_names.remove('v_thresh')
                parameter_names = ['v_thresh'] + parameter_names
            for cell in cell_list:
                new_parameters = {}
                for name in parameter_names:
                    if name == 'spike_times':
                        new_parameters[name] = [1.0, 2.0]
                    elif name == 'v_thresh':
                        new_parameters[name] = numpy.random.uniform(-100, 100)
                    elif name == 'v_reset' or name == 'v_init':
                        if 'v_thresh' in parameter_names:
                            new_parameters[name] = new_parameters['v_thresh'] - numpy.random.uniform(0.1, 100)
                        else:
                            new_parameters[name] = numpy.random.uniform(0.1, 100)
                    elif name == 'v_spike':
                        new_parameters[name] = new_parameters['v_thresh'] + numpy.random.uniform(0.1, 100)
                    else:
                        new_parameters[name] = numpy.random.uniform(0.1, 100) # tau_refrac is always at least dt (=0.1)
                try:
                    cell.set_parameters(**new_parameters)
                except Exception, e:
                    raise Exception("%s. %s in %s" % (e, new_parameters, cell_class))
                retrieved_parameters = cell.get_parameters()
                self.assertEqual(set(new_parameters.keys()), set(retrieved_parameters.keys()))
                
                for name in new_parameters:
                    i = new_parameters[name]; o = retrieved_parameters[name]
                    if name != 'spike_times':
                        self.assertEqual(type(i), type(o), "%s: input: %s, output: %s" % (name, type(i), type(o)))
                        self.assertAlmostEqual(i, o,
                                               IDSetGetTest.decimal_places.get(name, IDSetGetTest.default_dp),
                                               "%s in %s: %s != %s" % (name, cell_class.__name__, i,o))
    
    def testGetCellClass(self):
        assert 'cellclass' in common.IDMixin.non_parameter_attributes
        for name, pop in self.populations.items():
            assert isinstance(pop[0], common.IDMixin)
            assert 'cellclass' in pop[0].non_parameter_attributes
            self.assertEqual(pop[0].cellclass.__name__, name)
        self.assertRaises(Exception, setattr, pop[0].cellclass, 'dummy')
        
    def testGetSetPosition(self):
        for cell_group in self.cells.values():
            pos = cell_group[0].position
            self.assertEqual(len(pos), 3)
            cell_group[0].position = (9.8, 7.6, 5.4)
            self.assertEqual(tuple(cell_group[0].position), (9.8, 7.6, 5.4))

# ==============================================================================
class SetValueTest(unittest.TestCase):
    
    def setUp(self):
        self.cells = sim.create(sim.IF_curr_exp, n=10)
        self.single_cell = sim.create(sim.IF_cond_exp, n=1)
        
    def testSetFloat(self):
        sim.set(self.cells, 'tau_m',35.7)
        sim.set(self.single_cell, 'v_init', -67.8)
        for cell in self.cells:
            try:
                assert cell.tau_m == 35.7
            except AttributeError: # if cell is not on this node
                pass
        assert self.single_cell.v_init == -67.8

    def testSetDict(self):
        sim.set(self.cells, {'tau_m': 35.7, 'tau_syn_E': 5.432})
        for cell in self.cells:
            try:
                assert cell.tau_syn_E == 5.432
                assert cell.tau_m == 35.7
            except AttributeError: # if cell is not on this node
                pass
            
    def testSetNonExistentParameter(self):
        # note that although syn_shape is added to the NEURON parameter dict when creating
        # an IF_curr_exp, it is not a valid parameter to be changed later.
        self.assertRaises(common.NonExistentParameterError, sim.set, self.cells, 'syn_shape', 'alpha')
    
# ==============================================================================
class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
        
    def testSimpleInit(self):
        net = sim.Population((3,3), sim.IF_curr_alpha)
        self.assertEqual(net.size, 9)
        n_cells_local = len([id for id in net])
        # round-robin distribution
        min = 9/sim.num_processes()
        max = min+1
        assert min <= n_cells_local <= max, "%d not between %d and %d" % (n_cells_local, min, max)
    
    def testInitWithParams(self):
        net = sim.Population((3,3), sim.IF_curr_alpha, {'tau_syn_E':3.141592654})
        for cell in net:
            self.assertAlmostEqual(cell.tau_syn_E, 3.141592654, places=5)
    
    def testInitWithLabel(self):
        net = sim.Population((3,3), sim.IF_curr_alpha, label='iurghiushrg')
        assert net.label == 'iurghiushrg'
    
    def testInvalidCellType(self):
        self.assertRaises(common.InvalidModelError, sim.Population, (3,3), 'qwerty', {})
        
# ==============================================================================
class PopulationIndexTest(unittest.TestCase):
    """Tests of the Population class indexing."""
    
    def setUp(self):
        self.net1 = sim.Population((10,), sim.IF_curr_alpha)
        self.net2 = sim.Population((2,4,3), sim.IF_curr_exp)
        self.net3 = sim.Population((2,2,1), sim.SpikeSourceArray)
        self.net4 = sim.Population((1,2,1), sim.SpikeSourceArray)
        self.net5 = sim.Population((3,3), sim.IF_cond_alpha)
    
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
        self.net1 = sim.Population((10,), sim.IF_curr_alpha)
        self.net2 = sim.Population((2,4,3), sim.IF_curr_exp)
        self.net3 = sim.Population((2,2,1), sim.SpikeSourceArray)
        self.net4 = sim.Population((1,2,1), sim.SpikeSourceArray)
        self.net5 = sim.Population((3,3), sim.IF_cond_alpha)
        
    def testIter(self):
        """This needs more thought for the distributed case."""
        for net in self.net1, self.net2:
            ids = [i for i in net]
            self.assertEqual(ids, net.local_cells.tolist())
            self.assert_(isinstance(ids[0], sim.simulator.ID))
            
    def testAddressIter(self):
        for net in self.net1, self.net2:
            for id, addr in zip(net.ids(), net.addresses()):
                self.assertEqual(id, net[addr])
                self.assertEqual(addr, net.locate(id))
        
# ==============================================================================
class PopulationSpikesTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        self.spiketimes = numpy.arange(5,105,10.0)
        spiketimes_2D = self.spiketimes.reshape((len(self.spiketimes),1))
        self.input_spike_array = numpy.concatenate((numpy.zeros(spiketimes_2D.shape, 'float'), spiketimes_2D),
                                                   axis=1)
        self.p1 = sim.Population(1, sim.SpikeSourceArray, {'spike_times': self.spiketimes})
    
    def tearDown(self):
        pass
    
    def testGetSpikes(self):
        self.p1.record()
        sim.run(100.0)
        output_spike_array = self.p1.getSpikes()
        assert_arrays_almost_equal(self.input_spike_array, output_spike_array, 1e-11)
    
    def testPopulationRecordTwice(self):
        """Neurons should not be recorded twice.
        Multiple calls to `Population.record()` are ok, but a given neuron will only be
        recorded once."""
        self.p1.record()
        self.p1.record()
        sim.run(100.0)
        output_spike_array = self.p1.getSpikes()
        self.assertEqual(self.input_spike_array.shape, (10,2))
        self.assertEqual(self.input_spike_array.shape, output_spike_array.shape)

#===============================================================================
class PopulationSetTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        cell_params = {
            'tau_m' : 20.,  'tau_syn_E' : 2.3,   'tau_syn_I': 4.5,
            'v_rest': -55., 'v_reset'   : -62.3, 'v_thresh' : -50.2,
            'cm'    : 1.,   'tau_refrac': 2.3}
        self.p1 = sim.Population((5,4,3), sim.IF_curr_exp, cell_params)
        self.p2 = sim.Population((2,2), sim.SpikeSourceArray)
    
    def testSetFromDict(self):
        """Population.set()"""
        self.p1.set({'tau_m': 43.21})
        for cell in self.p1:
            assert cell.tau_m == 43.21
    
    def testSetFromPair(self):
        """Population.set(): A parameter set as a string, value pair should be retrievable using the top-level HocObject"""
        self.p1.set('tau_m', 12.34)
        for cell in self.p1:
            assert cell.tau_m == 12.34
        
    def testSetOnlyChangesTheDesiredParameters(self):
        before = [cell.get_parameters() for cell in self.p1]
        self.p1.set('v_init', -78.9)
        after = [cell.get_parameters() for cell in self.p1]
        for name in self.p1.celltype.__class__.default_parameters:
            if name == 'v_init':
                for a in after:
                    self.assertAlmostEqual(a[name], -78.9, places=5)
            else:
                for b,a in zip(before,after):
                    self.assert_(b[name] == a[name], "%s: %s != %s" % (name, b[name], a[name]))
                
    def test_set_invalid_type(self):
        self.assertRaises(common.InvalidParameterValueError, self.p1.set, 'foo', {})
        self.assertRaises(common.InvalidParameterValueError, self.p1.set, [1,2,3])
                
    def testSetInvalidFromDict(self):
        self.assertRaises(common.InvalidParameterValueError, self.p1.set, {'v_thresh':'hello','tau_m':56.78})
            
    def testSetNonexistentFromPair(self):
        """Population.set(): Trying to set a nonexistent parameter should raise an exception."""
        self.assertRaises(common.NonExistentParameterError, self.p1.set, 'tau_foo', 10.0)
    
    def testSetNonexistentFromDict(self):
        """Population.set(): When some of the parameters in a dict are inexistent, an exception should be raised.
           There is no guarantee that the existing parameters will be set."""
        self.assertRaises(common.NonExistentParameterError, self.p1.set, {'tau_foo': 10.0, 'tau_m': 21.0})
            
    def testRandomInit(self):
        rd = random.RandomDistribution('uniform', [-75,-55])
        self.p1.randomInit(rd)
        self.assertNotEqual(self.p1[0,0,0].v_init, self.p1[0,0,1].v_init)
                
    def test_tset(self):
        tau_m = numpy.arange(10.0, 16.0, 0.1).reshape((5,4,3))
        self.p1.tset("tau_m", tau_m)
        self.assertEqual(self.p1[0,0,0].tau_m, 10.0)
        self.assertEqual(self.p1[0,0,1].tau_m, 10.1)
        self.assertAlmostEqual(self.p1[0,3,1].tau_m, 11.0, 9)
        
        spike_times = numpy.arange(40.0).reshape(2,2,10)
        self.p2.tset("spike_times", spike_times)
        self.assertEqual(list(self.p2[0,0].spike_times), numpy.arange(10.0).tolist())
        self.assertEqual(list(self.p2[1,1].spike_times), numpy.arange(30.0,40.0).tolist())
                
    def testTSetInvalidDimensions(self):
        """Population.tset(): If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.assertRaises(common.InvalidDimensionsError, self.p1.tset, 'i_offset', array_in)
    
    #def testTSetInvalidValues(self):
    #    """Population.tset(): If some of the values in the valueArray are invalid, should raise an exception."""
    #    array_in = numpy.array([['potatoes','carrots'],['oranges','bananas']])
    #    self.assertRaises(common.InvalidParameterValueError, self.p2.tset, 'spike_times', array_in)
        
    def testRSetNumpy(self):
        """Population.rset(): with numpy rng."""
        rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.p1.rset('cm', rd1)
        output_values = self.p1.get('cm', as_array=True)
        input_values = rd2.next(len(self.p1)).reshape(self.p1.dim)
        assert_arrays_almost_equal(input_values, output_values, 1e-12)
                
#===============================================================================                
class PopulationPositionsTest(unittest.TestCase):
    
    def test_nearest(self):
        p = sim.Population((4,5,6), sim.IF_cond_exp)
        self.assertEqual(p.nearest((0.0,0.0,0.0)), p[0,0,0])
        self.assertEqual(p.nearest((0.0,1.0,0.0)), p[0,1,0])
        self.assertEqual(p.nearest((1.0,0.0,0.0)), p[1,0,0])
        self.assertEqual(p.nearest((3.0,2.0,1.0)), p[3,2,1])
        self.assertEqual(p.nearest((3.49,2.49,1.49)), p[3,2,1])
        self.assertEqual(p.nearest((3.49,2.49,1.51)), p[3,2,2])
        self.assertEqual(p.nearest((3.49,2.49,1.5)), p[3,2,2])
        self.assertEqual(p.nearest((2.5,2.5,1.5)), p[3,3,2])
                
#===============================================================================
class PopulationCellAccessTest(unittest.TestCase):
    
    def test_index(self):
        p = sim.Population((4,5,6), sim.IF_cond_exp)
        self.assertEqual(p.index(0), p[0,0,0])
        self.assertEqual(p.index(119), p[3,4,5])
        self.assertEqual(p.index([0,1,2]).tolist(), [p[0,0,0], p[0,0,1], p[0,0,2]])
     
# ==============================================================================
class PopulationRecordTest(unittest.TestCase): # to write later
    """Tests of the record(), record_v(), printSpikes(), print_v() and
       meanSpikeCount() methods of the Population class."""
    
    def setUp(self):
        self.pop1 = sim.Population((3,3), sim.SpikeSourcePoisson,{'rate': 20})
        self.pop2 = sim.Population((3,3), sim.IF_curr_alpha)
        #self.pop3 = sim.Population((3,3), sim.EIF_cond_alpha_isfa_ista)
        self.pops =[self.pop1, self.pop2] #, self.pop3]

    def tearDown(self):
        if hasattr(sim.simulator, 'reset'):
            sim.simulator.reset()

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
        #self.pop3.record()
        simtime = 1000.0
        sim.run(simtime)
        rate = self.pop1.meanSpikeCount()*1000/simtime
        if sim.rank() == 0: # only on master node
            assert (20*0.8 < rate) and (rate < 20*1.2), "rate is %s" % rate
        #rate = self.pop3.meanSpikeCount()*1000/simtime
        #self.assertEqual(rate, 0.0)
    
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
        sim.running = False
        sim.run(simtime)
        self.pop2.print_v("temp_neuron.v", gather=True, compatible_output=True)
    
    def testRecordWithSpikeTimesGreaterThanSimTime(self):
        """
        If a `SpikeSourceArray` is initialized with spike times greater than the
        simulation time, only those spikes that actually occurred should be
        written to file or returned by getSpikes().
        """
        spike_times = numpy.arange(10.0, 200.0, 10.0)
        spike_source = sim.Population(1, sim.SpikeSourceArray, {'spike_times': spike_times})
        spike_source.record()
        sim.run(100.0)
        spikes = spike_source.getSpikes()
        spikes = spikes[:,1]
        if sim.rank() == 0:
            self.assert_( max(spikes) == 100.0, str(spikes) )

                
#===============================================================================
class SynapticPlasticityTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
    
    def test_ProjectionInit(self):
        for wd in (sim.AdditiveWeightDependence(),
                   sim.MultiplicativeWeightDependence(),
                   sim.AdditivePotentiationMultiplicativeDepression()):
            fast_mech = sim.TsodyksMarkramMechanism()
            slow_mech = sim.STDPMechanism(
                        timing_dependence=sim.SpikePairRule(),
                        weight_dependence=wd,
                        dendritic_delay_fraction=1.0
            )
            p1 = sim.Population(10, sim.SpikeSourceArray)
            p2 = sim.Population(10, sim.IF_cond_exp)
            prj1 = sim.Projection(p1, p2, sim.OneToOneConnector(),
                                  synapse_dynamics=sim.SynapseDynamics(fast_mech, None))
            prj2 = sim.Projection(p1, p2, sim.OneToOneConnector(),
                                  synapse_dynamics=sim.SynapseDynamics(None, slow_mech))
                
#===============================================================================
class ProjectionTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        p1 = sim.Population(10, sim.SpikeSourceArray)
        p2 = sim.Population(10, sim.IF_cond_exp)
        self.prj = sim.Projection(p1, p2, sim.OneToOneConnector())
        
    def test_describe(self):
        self.prj.describe()
        
    def test_printWeights(self):
        self.prj.printWeights("weights_list.tmp", format='list', gather=False)
        self.prj.printWeights("weights_array.tmp", format='array', gather=False)
        # test needs completing. Should read in the weights and check they have the correct values
         
         
#===============================================================================
class ConnectorsTest(unittest.TestCase):
    
    def test_OneToOne_with_unequal_pop_sizes(self):
        sim.setup()
        p1 = sim.Population(10, sim.SpikeSourceArray)
        p2 = sim.Population(9, sim.IF_cond_exp)
        c = sim.OneToOneConnector()
        self.assertRaises(Exception, sim.Projection, p1, p2, c) 
                
#===============================================================================
class ElectrodesTest(unittest.TestCase):
    
    def test_DCSource(self):
        # just check no Exceptions are raised, for now.
        source = sim.DCSource(amplitude=0.5, start=50.0, stop=100.0)
        cells = sim.create(sim.IF_curr_exp, {}, 5)
        source.inject_into(cells)
        for cell in cells:
            cell.inject(source)
                
    def test_StepCurrentSource(self):
        # just check no Exceptions are raised, for now.
        source = sim.StepCurrentSource([10.0, 20.0, 30.0, 40.0], [-0.1, 0.2, -0.1, 0.0])
        cells = sim.create(sim.IF_curr_exp, {}, 5)
        source.inject_into(cells)
        for cell in cells:
            cell.inject(source)
                
#===============================================================================
class StateTest(unittest.TestCase):
    
    def test_get_time(self):
        sim.setup()
        self.assertEqual(sim.get_current_time(), 0.0)
        sim.run(100.0)
        self.assertAlmostEqual(sim.get_current_time(), 100.0, 9)
        
    def test_get_time_step(self):
        sim.setup()
        self.assertEqual(sim.get_time_step(), 0.1)
        sim.setup(timestep=0.05)
        self.assertEqual(sim.get_time_step(), 0.05)
                
# ==============================================================================
if __name__ == "__main__":
    simulator = sys.argv[1]
    sys.argv.remove(simulator) # because unittest.main() processes sys.argv
    sim = __import__("pyNN.%s" % simulator, None, None, [simulator])
    sim.setup()
    unittest.main()