"""
Unit tests for all simulators
$Id$
"""

import sys
import unittest
import numpy
import os
import cPickle as pickle
from pyNN import common, random, utility, recording, errors, space
import glob

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def assert_arrays_almost_equal(a, b, threshold, msg=''):
    if a.shape != b.shape:
        raise unittest.TestCase.failureException("Shape mismatch: a.shape=%s, b.shape=%s" % (a.shape, b.shape))
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        if msg:
            err_msg += "\nOther information: %s" % msg
        raise unittest.TestCase.failureException(err_msg)

def is_local(cell):
    return True # implement this for testing distributed simulation

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def setUp(self):
        sim.setup()
    
    def testCreateStandardCell(self):
        for cellclass in sim.list_standard_models():
            ifcell = sim.create(cellclass)
            assert isinstance(ifcell, common.IDMixin), type(ifcell)
        
    def testCreateStandardCells(self):
        for cellclass in sim.list_standard_models():
            ifcells = sim.create(cellclass, n=10)
            self.assertEqual(len(ifcells), 10)
            local_cells = [cell for cell in ifcells if cell.local]
            if hasattr(cellclass, 'always_local') and cellclass.always_local:
                cells_per_process = 10.0
            else:
                cells_per_process = 10.0/sim.num_processes()
            self.assert_(numpy.floor(cells_per_process) <= len(local_cells) <=  numpy.ceil(cells_per_process), "cells per process: %d, local cells: %d" % (cells_per_process, len(local_cells)))
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""
        self.assertRaises(AssertionError, sim.create, sim.IF_curr_alpha, n=-1)

    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with the top-level HocObject"""
        ifcell = sim.create(sim.IF_curr_alpha,{'tau_syn_E':3.141592654})
        if ifcell.local:
            self.assertAlmostEqual(ifcell.tau_syn_E, 3.141592654, places=6)
        
    def testCreateInvalidCell(self):
        """create(): Trying to create a cell type which is not a standard cell or
        valid native cell should raise an Exception."""
        self.assertRaises(errors.InvalidModelError, sim.create, 'qwerty', n=10)
    
    def testCreateWithInvalidParameter(self):
        """create(): Creating a cell with an invalid parameter should raise an Exception."""
        self.assertRaises(errors.NonExistentParameterError, sim.create, sim.IF_curr_alpha, {'tau_foo':3.141592654})  

# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        sim.setup()
        self.precells = sim.create(sim.SpikeSourcePoisson, n=7)
        self.postcells = sim.create(sim.IF_curr_alpha, n=5)
        
    def testConnectTwoCells(self):
        conn_list = sim.connect(self.precells[0], self.postcells[0])
        if self.postcells[0].local:
            self.assertEqual(len(conn_list), 1)
        else:
            self.assertEqual(len(conn_list), 0)
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""
        conn_list = sim.connect(self.precells[0], self.postcells[0], weight=0.1234)
        if conn_list:
            weight = conn_list[0].weight
            self.assertAlmostEqual(weight, 0.1234, 6)
            
    def testConnectTwoCellsWithDelay(self):
        conn_list = sim.connect(self.precells[0], self.postcells[0], delay=4.321)
        if conn_list:
            delay = conn_list[0].delay
            if sim_name == 'nest':
                self.assertEqual(round(delay, 1), 4.4) # NEST rounds delays to the timestep, 0.1 here
            else:
                self.assertAlmostEqual(delay, 4.321, 6) 
    
    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n,
        each element being the id number of a netcon."""
        conn_list = sim.connect(self.precells, self.postcells[0])
        # connections are only created on the node containing the post-syn
        if self.postcells[0].local:
            self.assertEqual(len(conn_list), len(self.precells))
        else:
            self.assertEqual(len(conn_list), 0)
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        conn_list = sim.connect(self.precells[0], self.postcells)
        cells_on_this_node = len([i for i in self.postcells if i.local])
        self.assertEqual(len(conn_list),  cells_on_this_node)
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        conn_list = sim.connect(self.precells, self.postcells)
        cells_on_this_node = len([i for i in self.postcells if i.local])
        expected_length = cells_on_this_node*len(self.precells)
        self.assertEqual(len(conn_list), expected_length, "%d != %d" % (len(conn_list), expected_length))
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        conn_list = sim.connect(self.precells, self.postcells, p=0.5)
        cells_on_this_node = len([i for i in self.postcells if i.local])
        assert 0 < len(conn_list) < len(self.precells)*cells_on_this_node, 'Number of connections is %d: this is very unlikely (although possible).' % len(conn_list)
    
    def testConnectNonExistentPreCell(self):
        """connect(): Connecting from non-existent cell should raise a ConnectionError."""
        if self.postcells[0].local:
            self.assertRaises(errors.ConnectionError, sim.connect, 12345, self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        self.assertRaises(errors.ConnectionError, sim.connect, self.precells[0], 'cell45678')
    
    def testInvalidSourceId(self):
        """connect(): sources must be integers."""
        self.precells.append('74367598')
        self.assertRaises(errors.ConnectionError, sim.connect, self.precells, self.postcells)
    
    def testInvalidTargetId(self):
        """connect(): targets must be integers."""
        self.postcells.append('99.9')
        self.assertRaises(errors.ConnectionError, sim.connect, self.precells, self.postcells)
    
    def testConnectTooSmallDelay(self):
        self.assertRaises(errors.ConnectionError, sim.connect, self.precells[0], self.postcells[0], delay=1e-12)

# ==============================================================================
class IDSetGetTest(unittest.TestCase):
    """Tests of the ID.__setattr__()`, `ID.__getattr()` `ID.setParameters()`
    and `ID.getParameters()` methods for all available standard cell types
    and for both lone and in-population IDs."""
    
    model_list = []
    default_dp = 4
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
    
    def testSetGet(self):
        """__setattr__(), __getattr__(): sanity check"""
        for cell_class in IDSetGetTest.model_list:
            cell_list = [cell for cell in self.cells[cell_class.__name__] if cell.local] + \
                        [cell for cell in self.populations[cell_class.__name__].local_cells]
            parameter_names = cell_class.default_parameters.keys()
            for cell in cell_list:
                for name in parameter_names:
                    if name == 'spike_times':
                        i = [1.0, 2.0]
                        cell.__setattr__(name, i)
                        o = list(cell.__getattr__(name))
                        assert isinstance(i, list), type(i)
                        assert isinstance(o, list), type(o)
                        try:
                            assert i == o, "%s (%s) != %s (%s)" % (i,o, type(i), type(o))
                        except ValueError, errmsg:
                            raise ValueError("%s. %s (%s) != %s (%s)" % (errmsg, i, type(i), o, type(o)))
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
            cell_list = [cell for cell in self.cells[cell_class.__name__] if cell.local] + \
                        [cell for cell in self.populations[cell_class.__name__].local_cells]
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
            if len(pop.local_cells)>0:
                self.assertEqual(pop.local_cells[0].cellclass.__name__, name)
        if len(pop.local_cells)>0:
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
        sim.setup()
        self.cells = sim.create(sim.IF_curr_exp, n=10)
        self.single_cell = sim.create(sim.IF_cond_exp, n=1)
        
    def testSetFloat(self):
        sim.set(self.cells, 'tau_m',35.7)
        sim.set(self.single_cell, 'v_init', -67.8)
        for cell in self.cells:
            try:
                self.assertAlmostEqual(cell.tau_m, 35.7, 5)
            except errors.NotLocalError: # if cell is not on this node
                pass
        if self.single_cell.local:
            self.assertAlmostEqual(self.single_cell.v_init, -67.8, 6)

    def testSetDict(self):
        sim.set(self.cells, {'tau_m': 35.7, 'tau_syn_E': 5.432})
        for cell in self.cells:
            try:
                self.assertAlmostEqual(cell.tau_syn_E, 5.432, 6)
                self.assertAlmostEqual(cell.tau_m, 35.7, 5)
            except errors.NotLocalError: # if cell is not on this node
                pass
            
    def testSetNonExistentParameter(self):
        # note that although syn_shape is added to the NEURON parameter dict when creating
        # an IF_curr_exp, it is not a valid parameter to be changed later.
        self.assertRaises(errors.NonExistentParameterError, sim.set, self.cells, 'syn_shape', 'alpha')

    def testSetZero(self):
        sim.set(self.cells, 'v_thresh', 0.0)
    
# ==============================================================================
class PopulationInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Population class."""
    
    def setUp(self):
        sim.setup()
        
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
        self.assertRaises(errors.InvalidModelError, sim.Population, (3,3), 'qwerty', {})
        
# ==============================================================================
class PopulationIndexTest(unittest.TestCase):
    """Tests of the Population class indexing."""
    
    def setUp(self):
        sim.setup()
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
        self.assertRaises(errors.InvalidDimensionsError, self.net1.__getitem__, (10,2))
        
# ==============================================================================
class PopulationIteratorTest(unittest.TestCase):
    """Tests of the Population class iterators."""
    
    def setUp(self):
        sim.setup()
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
            self.assert_(isinstance(ids[0], common.IDMixin))
            
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
        #spiketimes_2D = self.spiketimes.reshape((len(self.spiketimes),1))
        #self.input_spike_array = numpy.concatenate((numpy.zeros(spiketimes_2D.shape, 'float'), spiketimes_2D),
        #                                           axis=1)
        self.input_spike_array = self.spiketimes
        self.p1 = sim.Population(1, sim.SpikeSourceArray, {'spike_times': self.spiketimes})
    
    def tearDown(self):
        pass
    
    def testGetSpikes(self):
        self.p1.record()
        sim.run(100.0)
        output_spike_array = self.p1.getSpikes()[:,1]
        if sim.rank() == 0:
            assert_arrays_almost_equal(self.input_spike_array, output_spike_array, 1e-11)
    
    def testPopulationRecordTwice(self):
        """Neurons should not be recorded twice.
        Multiple calls to `Population.record()` are ok, but a given neuron will only be
        recorded once."""
        self.p1.record()
        self.p1.record()
        sim.run(100.0)
        output_spike_array = self.p1.getSpikes()
        self.assertEqual(self.input_spike_array.shape, (10,))
        if sim.rank() == 0:
            self.assertEqual(self.input_spike_array.size, output_spike_array.shape[0])

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
            self.assertAlmostEqual(cell.tau_m, 43.21, 6)
    
    def testSetFromPair(self):
        """Population.set(): A parameter set as a string, value pair should be retrievable using the top-level HocObject"""
        self.p1.set('tau_m', 12.34)
        for cell in self.p1:
            self.assertAlmostEqual(cell.tau_m, 12.34, 6)
        
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
        self.assertRaises(errors.InvalidParameterValueError, self.p1.set, 'foo', {})
        self.assertRaises(errors.InvalidParameterValueError, self.p1.set, [1,2,3])
                
    def testSetInvalidFromDict(self):
        self.assertRaises(errors.InvalidParameterValueError, self.p1.set, {'v_thresh':'hello','tau_m':56.78})
            
    def testSetNonexistentFromPair(self):
        """Population.set(): Trying to set a nonexistent parameter should raise an exception."""
        self.assertRaises(errors.NonExistentParameterError, self.p1.set, 'tau_foo', 10.0)
    
    def testSetNonexistentFromDict(self):
        """Population.set(): When some of the parameters in a dict are inexistent, an exception should be raised.
           There is no guarantee that the existing parameters will be set."""
        self.assertRaises(errors.NonExistentParameterError, self.p1.set, {'tau_foo': 10.0, 'tau_m': 21.0})
            
    def testRandomInit(self):
        rd = random.RandomDistribution('uniform', [-75,-55])
        self.p1.randomInit(rd)
        #self.assertNotEqual(self.p1[0,0,0].v_init, self.p1[0,0,1].v_init)
        self.assertNotEqual(self.p1.local_cells[0].v_init, self.p1.local_cells[1].v_init)
                
    def test_tset(self):
        tau_m = numpy.arange(10.0, 16.0, 0.1).reshape((5,4,3))
        self.p1.tset("tau_m", tau_m)
        
        for addr, val in ( ((0,0,0), 10.0),
                           ((0,0,1), 10.1),
                           ((0,3,1), 11.0)):
            if self.p1[addr].local:
                self.assertAlmostEqual(self.p1[addr].tau_m, val, 6)
        
        spike_times = numpy.arange(40.0).reshape(2,2,10)
        self.p2.tset("spike_times", spike_times)
        if self.p2[0,0].local:
            assert_arrays_almost_equal(self.p2[0,0].spike_times, numpy.arange(10.0), 1e-9)
        if self.p2[1,1].local:
            assert_arrays_almost_equal(self.p2[1,1].spike_times, numpy.arange(30.0,40.0), 1e-9)
                
    def testTSetInvalidDimensions(self):
        """Population.tset(): If the size of the valueArray does not match that of the Population, should raise an InvalidDimensionsError."""
        array_in = numpy.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.assertRaises(errors.InvalidDimensionsError, self.p1.tset, 'i_offset', array_in)
    
    def testTSetInvalidValues(self):
        """Population.tset(): If some of the values in the valueArray are invalid, should raise an exception."""
        array_in = numpy.array([['potatoes','carrots'],['oranges','bananas']])
        self.assertRaises(errors.InvalidParameterValueError, self.p2.tset, 'spike_times', array_in)
        
    def testRSetNumpy(self):
        """Population.rset(): with numpy rng."""
        rd1 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765, num_processes=sim.num_processes()),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        rd2 = random.RandomDistribution(rng=random.NumpyRNG(seed=98765, num_processes=sim.num_processes()),
                                         distribution='uniform',
                                         parameters=[0.9,1.1])
        self.p1.rset('cm', rd1)
        output_values = self.p1.get('cm', as_array=True)
        mask = (1-numpy.isnan(output_values)).astype(bool)
        input_values = rd2.next(len(self.p1), mask_local=False).reshape(self.p1.dim)
        assert_arrays_almost_equal(input_values[mask], output_values[mask], 1e-7)
                
#===============================================================================                
class PopulationPositionsTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
    
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
    
    def setUp(self):
        sim.setup()
        
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
        sim.setup()
        self.pop1 = sim.Population((3,3), sim.SpikeSourcePoisson,{'rate': 20})
        self.pop2 = sim.Population((3,3), sim.IF_curr_alpha)
        self.pop3 = sim.Population((3,3), sim.IF_cond_alpha)
        #self.pop4 = sim.Population((3,3), sim.EIF_cond_alpha_isfa_ista)
        self.pops = [self.pop1, self.pop2, self.pop3]

    def tearDown(self):
        sim.end()

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
        msc = self.pop1.meanSpikeCount()
        if sim.rank() == 0: # only on master node
            rate = msc*1000/simtime
            ##print self.pop1.recorders['spikes'].recorders
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
        cells_to_record = [self.pop2[0,0], self.pop2[1,1]]
        self.pop2.record_v(cells_to_record)
        simtime = 10.0
        sim.run(simtime)
        self.pop2.print_v("temp.v", gather=True, compatible_output=True)
        vm = self.pop2.get_v()
        if sim.rank() == 0:
            self.assertEqual(vm.shape, ((1+int(10.0/0.1))*len(cells_to_record), 3))
            vm_fromfile = numpy.loadtxt("temp.v")
            #print vm_fromfile
            self.assertEqual(vm_fromfile.shape, ((1+int(10.0/0.1))*len(cells_to_record), 2))
            os.remove("temp.v")
        
    def testSynapticConductanceRecording(self):
        # current-based synapses
        self.assertRaises(errors.RecordingError, self.pop2.record_gsyn)
        # conductance-based synapses
        cells_to_record = [self.pop3[1,0], self.pop3[2,2]]
        self.pop3.record_gsyn(cells_to_record)
        simtime = 10.0
        sim.run(simtime)
        gsyn = self.pop3.get_gsyn()
        if sim.rank() == 0:
            self.assertEqual(gsyn.shape, ((1+int(10.0/0.1))*len(cells_to_record), 4))
    
    def testRecordWithSpikeTimesGreaterThanSimTime(self):
        """
        If a `SpikeSourceArray` is initialized with spike times greater than the
        simulation time, only those spikes that actually occurred should be
        written to file or returned by getSpikes().
        """
        spike_times = numpy.arange(10.0, 200.0, 10.0)
        spike_source = sim.Population(1, sim.SpikeSourceArray, {'spike_times': spike_times})
        spike_source.record()
        sim.run(101.0)
        spikes = spike_source.getSpikes()
        spikes = spikes[:,1]
        if sim.rank() == 0:
            self.assertAlmostEqual(max(spikes), 100.0, 6)

    def testRecordVmFromSpikeSource(self):
        self.assertRaises(errors.RecordingError, self.pop1.record_v)
        
    
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
        for filename in "weights_list.tmp", "weights_array.tmp":
            if os.path.exists(filename):
                os.remove(filename)
         
    def test_iterator(self):
        assert hasattr(self.prj, "connections")
        assert not callable(self.prj.connections)
        assert hasattr(self.prj.connections, "__iter__")
        connections = [c for c in self.prj.connections]
        self.assertEqual(len(connections), len(self.prj.post.local_cells))
         
# ==============================================================================
class ProjectionInitTest(unittest.TestCase):
    """Tests of the __init__() method of the Projection class."""
        
    def setUp(self):
        sim.setup()
        sim.Population.nPop = 0
        sim.Projection.nProj = 0
        self.target33    = sim.Population((3,3), sim.IF_curr_alpha, label="target33")
        self.target6     = sim.Population((6,), sim.IF_curr_alpha, label="target6")
        self.target1     = sim.Population((1,), sim.IF_cond_exp, label="target1")
        self.source5     = sim.Population((5,), sim.SpikeSourcePoisson, label="source5")
        self.source22    = sim.Population((2,2), sim.SpikeSourcePoisson, label="source22")
        self.source33    = sim.Population((3,3), sim.SpikeSourcePoisson, label="source33")
        self.expoisson33 = sim.Population((3,3), sim.SpikeSourcePoisson,{'rate': 100}, label="expoisson33")
        
    def testAllToAll(self):
        for srcP in [self.source5, self.source22, self.target33]:
            for tgtP in [self.target6, self.target33]:
                prj = sim.Projection(srcP, tgtP, sim.AllToAllConnector(allow_self_connections=False))
                prj.setWeights(1.234)
                weights = []
                for c in prj.connections:
                    weights.append(c.weight)
                if srcP==tgtP:
                    n = (len(srcP)-1)*len(tgtP.local_cells)                    
                else:
                    n = len(srcP)*len(tgtP.local_cells)
                self.assertEqual(len(weights), n)
                target_weights = [1.234]*n
                assert_arrays_almost_equal(numpy.array(weights), numpy.array(target_weights), 1e-7) #msg="srcP=%s, tgtP=%s\n%s !=\n%s" % (srcP.label, tgtP.label, weights, target_weights))
            
    def testFixedProbability(self):
        """For all connections created with "fixedProbability"..."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target1, self.target6, self.target33]:
                prj = sim.Projection(srcP, tgtP, sim.FixedProbabilityConnector(0.5), rng=random.NumpyRNG(12345, rank=sim.rank(), num_processes=sim.num_processes()))
                if len(tgtP.local_cells) > 1:
                    assert (0 < len(prj) < len(srcP)*len(tgtP.local_cells)), 'len(prj) = %d, len(srcP)*len(tgtP.local_cells) = %d' % (len(prj), len(srcP)*len(tgtP.local_cells))
                
    def testOneToOne(self):
        """For all connections created with "OneToOne" ..."""
        prj = sim.Projection(self.source33, self.target33, sim.OneToOneConnector())
        assert len(prj.connections) == len(self.target33.local_cells), prj.connections
     
    def testDistanceDependentProbability(self):
        """For all connections created with "distanceDependentProbability"..."""
        # Test should be improved..."
        for rngclass in (random.NumpyRNG,): # random.NativeRNG):
            for expr in ('exp(-d)', 'd < 0.5'):
                prj = sim.Projection(self.source33, self.target33,
                                        sim.DistanceDependentProbabilityConnector(d_expression=expr),
                                        rng=rngclass(12345, rank=sim.rank(), num_processes=sim.num_processes()))
                assert (0 < len(prj) < len(self.source33)*len(self.target33))
        self.assertRaises(ZeroDivisionError, sim.DistanceDependentProbabilityConnector, d_expression="d/0.0")
    
    def testFixedNumberPre(self):
        c1 = sim.FixedNumberPreConnector(10)
        c2 = sim.FixedNumberPreConnector(3)
        c3 = sim.FixedNumberPreConnector(random.RandomDistribution('poisson',[5]))
        c4 = sim.FixedNumberPreConnector(10, allow_self_connections=False)
        for srcP in [self.source5, self.source22, self.target33]:
            for tgtP in [self.target6, self.target33]:
                for c in c1, c2, c4:
                    prj1 = sim.Projection(srcP, tgtP, c)
                    self.assertEqual(len(prj1.connections), c.n*len(tgtP.local_cells))
                prj3 = sim.Projection(srcP, tgtP, c3) # just a test that no Exceptions are raised
        self.assertRaises(Exception, sim.FixedNumberPreConnector, None)
        
    def testFixedNumberPost(self):
        c1 = sim.FixedNumberPostConnector(10)
        c2 = sim.FixedNumberPostConnector(3)
        c3 = sim.FixedNumberPostConnector(random.RandomDistribution('poisson',[5]))
        c4 = sim.FixedNumberPostConnector(10, allow_self_connections=False)
        for srcP in [self.source5, self.source22, self.target33]:
            for tgtP in [self.target6, self.target33]:
                for c in c1, c2, c4:
                    prj1 = sim.Projection(srcP, tgtP, c)
                    #print sim.rank(), c.n, len(srcP), c.n*len(srcP), len(prj1.connections)
                    if sim.num_processes() == 1:
                        self.assertEqual(len(prj1.connections), c.n*len(srcP))
                    else:
                        if MPI:
                            total_connections = MPI.COMM_WORLD.allreduce(len(prj1.connections), op=MPI.SUM)
                            self.assertEqual(total_connections, c.n*len(srcP))
                        else:
                            conn_per_node = float(c.n*len(srcP))/sim.num_processes()
                            self.assert_(0.8*conn_per_node < len(prj1.connections) < 1.2*conn_per_node+2, "len(connections)=%d, conn_per_node=%d prj=%s, n=%d" % (len(prj1.connections), conn_per_node, prj1.label, c.n) )
                    
                prj2 = sim.Projection(srcP, tgtP, c3) # just a test that no Exceptions are raised
        self.assertRaises(Exception, sim.FixedNumberPostConnector, None)
    
    def testFromList(self):
        connection_list = [
            ([0,], [0,], 0.1, 0.1),
            ([3,], [0,], 0.2, 0.11),
            ([2,], [3,], 0.3, 0.12),
            ([4,], [2,], 0.4, 0.13),
            ([0,], [1,], 0.5, 0.14),
            ]
        c1 = sim.FromListConnector(connection_list)
        prj = sim.Projection(self.source5, self.target6, c1)
        n_local = 0
        for src, tgt, w, d in connection_list:
            if prj.post[tuple(tgt)].local:
                n_local += 1
        self.assertEqual(len(prj.connections), n_local)
            
    def testSaveAndLoad(self):
        prj1 = sim.Projection(self.source33, self.target33, sim.OneToOneConnector())
        prj1.setDelays(1)
        prj1.setWeights(1.234)
        prj1.saveConnections("connections.tmp", gather=False)
        if sim.num_processes() > 1:
            distributed = True
        else:
            distributed = False
        prj2 = sim.Projection(self.source33, self.target33, sim.FromFileConnector("connections.tmp",
                                                                                  distributed=distributed))
        w1 = []; w2 = []; d1 = []; d2 = []
        # For a connections scheme saved and reloaded, we test if the connections, their weights and their delays
        # are equal.
        for c1,c2 in zip(prj1.connections, prj2.connections):
            w1.append(c1.weight)
            w2.append(c2.weight)
            d1.append(c1.delay)
            d2.append(c2.delay)
        assert (w1 == w2), 'w1 = %s\nw2 = %s' % (w1, w2)
        assert (d1 == d2), 'd1 = %s\nd2 = %s' % (d1, d2)
        if sim.rank() == 0:
            for filename in glob.glob("connections.tmp*"):
                os.remove(filename)
          
    def testSettingDelays(self):
        """Delays should be set correctly when using a Connector object."""
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                prj1 = sim.Projection(srcP, tgtP, sim.AllToAllConnector(delays=0.321))
                if sim_name != 'nest':
                    self.assertAlmostEqual(prj1.connections[0].delay, 0.321, 6)
                else:
                    self.assertAlmostEqual(prj1.connections[0].delay, 0.4, 6) # nest rounds delays to the timestep
         
    def testDistanceDependentWeights(self):
        connectors = (
            sim.AllToAllConnector(weights="exp(-d/10.0)", space=space.Space(scale_factor=0.9)),
            sim.FixedProbabilityConnector(0.8, weights="maximum(3-d, 0)", space=space.Space(offset=0.1)),
            sim.DistanceDependentProbabilityConnector("abs(d<3)", weights="sin(d)", space=space.Space(offset=0.1,
                                                                                                      scale_factor=0.9)),
        )
        exp = numpy.exp
        maximum = numpy.maximum
        sin = numpy.sin
        for srcP in [self.source5, self.source22]:
            for tgtP in [self.target6, self.target33]:
                for conn in connectors:
                    #print conn.w_expr
                    prj = sim.Projection(srcP, tgtP, conn)
                    first_connection = prj.connections[0]
                    last_connection = prj.connections[-1]
                    for c in first_connection, last_connection:
                       d = space.distance(c.source, c.target)
                       self.assertAlmostEqual(c.weight, eval(conn.weights), 10)

class ProjectionSetTest(unittest.TestCase):
    """Tests of the setWeights(), setDelays(), randomizeWeights() and
    randomizeDelays() methods of the Projection class."""

    def setUp(self):
        sim.setup()
        self.target_curr = sim.Population((3,3), sim.IF_curr_alpha)
        self.target_cond = sim.Population(5, sim.IF_cond_exp)
        self.targets = (self.target_curr, self.target_cond)
        self.source   = sim.Population((3,3), sim.SpikeSourcePoisson,{'rate': 200})
        self.distrib_Numpy = random.RandomDistribution(rng=random.NumpyRNG(12345), distribution='uniform', parameters=(0.2,1)) 
        self.distrib_Native= random.RandomDistribution(rng=random.NativeRNG(12345), distribution='uniform', parameters=(0.2,1)) 
        
    def testSetPositiveWeights(self):
        prj1 = sim.Projection(self.source, self.target_curr, sim.AllToAllConnector(), target='excitatory', label="exc, curr")
        prj2 = sim.Projection(self.source, self.target_curr, sim.AllToAllConnector(), target='inhibitory', label="inh, curr")
        prj3 = sim.Projection(self.source, self.target_cond, sim.AllToAllConnector(), target='excitatory', label="exc, cond")
        prj4 = sim.Projection(self.source, self.target_cond, sim.AllToAllConnector(), target='inhibitory', label="inh, cond")
        for prj in prj1, prj3, prj4:
            prj.setWeights(2.345)
            weights = []
            for c in prj.connections:
                weights.append(c.weight)
            result = 2.345*numpy.ones(len(prj.connections))
            assert_arrays_almost_equal(numpy.array(weights), result, 1e-7, msg=prj.label)
        self.assertRaises(errors.InvalidWeightError, prj2.setWeights, 2.345) # current-based inhibitory needs negative weights
            
    def testSetNegativeWeights(self):
        prj1 = sim.Projection(self.source, self.target_curr, sim.AllToAllConnector(), target='excitatory')
        prj2 = sim.Projection(self.source, self.target_curr, sim.AllToAllConnector(), target='inhibitory')
        prj3 = sim.Projection(self.source, self.target_cond, sim.AllToAllConnector(), target='excitatory')
        prj4 = sim.Projection(self.source, self.target_cond, sim.AllToAllConnector(), target='inhibitory')
        prj2.setWeights(-2.345)
        weights = []
        for c in prj2.connections:
            weights.append(c.weight)
        result = -2.345*numpy.ones(len(prj2.connections))
        assert_arrays_almost_equal(numpy.array(weights), result, 1e-7)
        for prj in prj1, prj3, prj4:
            self.assertRaises(errors.InvalidWeightError, prj.setWeights, -2.345) 
    
    def test_set_weights_with_array(self):
        prj = sim.Projection(self.source, self.target_curr,
                             sim.FixedProbabilityConnector(0.5, weights=self.distrib_Numpy),
                             target='excitatory')
        weight_array = prj.getWeights(format='array')
        prj.setWeights(weight_array + 0.5)
        def filter_NaN(arr):
            nan_filter = (1-numpy.isnan(arr)).astype(bool)
            return arr[nan_filter]
        assert_arrays_almost_equal(filter_NaN(weight_array + 0.5), filter_NaN(prj.getWeights(format='array')), 1e-7)
    
    def testSetDelays(self):
        for target in self.targets:
            prj1 = sim.Projection(self.source, target, sim.AllToAllConnector())
            prj1.setDelays(2.345)
            delays = []
            for c in prj1.connections:
                delays.append(c.delay)
            if sim_name != 'nest':
                result = 2.345*numpy.ones(len(prj1.connections))
            else:
                result = 2.4*numpy.ones(len(prj1.connections)) # nest rounds delays up
            assert_arrays_almost_equal(numpy.array(delays), result, 1e-7)
            
    def testRandomizeWeights(self):
        # The probability of having two consecutive weight vectors that are equal should be effectively 0
        for target in self.targets:
            prj1 = sim.Projection(self.source, target, sim.AllToAllConnector())
            prj1.randomizeWeights(self.distrib_Numpy)
            w1 = []; w2 = [];
            for c in prj1.connections:
                w1.append(c.weight)
            prj1.randomizeWeights(self.distrib_Numpy)        
            for c in prj1.connections:
                w2.append(c.weight)
            self.assertNotEqual(w1,w2)
            self.assertEqual(w2[0], prj1.connections[0].weight)
            
    def testRandomizeDelays(self):
        # The probability of having two consecutive delay vectors that are equal should be effectively 0
        for target in self.targets:
            prj1 = sim.Projection(self.source, target, sim.FixedProbabilityConnector(0.8))
            prj1.randomizeDelays(self.distrib_Numpy)
            d1 = []; d2 = [];
            for c in prj1.connections:
                d1.append(c.delay)
            prj1.randomizeDelays(self.distrib_Numpy)        
            for c in prj1.connections:
                d2.append(c.delay)
            self.assertNotEqual(d1,d2)
         
#===============================================================================
class ProjectionGetTest(unittest.TestCase):
    """Tests of the getWeights(), getDelays() methods of the Projection class."""

    def setUp(self):
        sim.setup(max_delay=0.5)
        sim.Population.nPop = 0
        self.target33 = sim.Population((3,3), sim.IF_curr_alpha, label="target33")
        self.target6  = sim.Population((6,), sim.IF_curr_alpha, label="target6")
        self.source5  = sim.Population((5,), sim.SpikeSourcePoisson, label="source5")
        self.source22 = sim.Population((2,2), sim.IF_curr_exp, label="source22")
        self.prjlist = []
        self.distrib_Numpy = random.RandomDistribution(rng=random.NumpyRNG(12345), distribution='uniform', parameters=(0.1,0.5))
        for tgtP in [self.target6, self.target33]:
            for srcP in [self.source5, self.source22]:
                for method in (sim.AllToAllConnector(), sim.FixedProbabilityConnector(p_connect=0.5)):
                    self.prjlist.append(sim.Projection(srcP,tgtP,method))

    def testGetWeightsWithList(self):
        for prj in self.prjlist:
            weights_in = self.distrib_Numpy.next(len(prj))
            prj.setWeights(weights_in)
            weights_out = numpy.array(prj.getWeights(format='list'))
            assert_arrays_almost_equal(weights_in, weights_out, 1e-7)
            
    def testGetWeightsWithArray(self):
        """Making 1D and removing weights <= 0 should turn the array format of getWeights()
        into the list format."""
        for prj in self.prjlist: 
            weights_in = self.distrib_Numpy.next(len(prj))
            prj.setWeights(weights_in)
            weights_out = numpy.array(prj.getWeights(format='array')).flatten()
            weights_out = weights_out.compress(weights_out>0)
            if weights_out.size > 0:
                self.assertEqual(weights_out[0], prj.connections[0].weight)
            self.assertEqual(weights_in.shape, weights_out.shape)
            assert_arrays_almost_equal(weights_in, weights_out, 1e-7)
            
         
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
    
    def setUp(self):
        sim.setup()
        
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
                
    # what should happen if we inject multiple electrodes into a single cell?
                
#===============================================================================
class StateTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        self.cells = sim.create(sim.IF_curr_exp) # brian chokes if there are no NeuronGroups in the Network object when net.run() is called
    
    def test_get_time(self):
        self.assertEqual(sim.get_current_time(), 0.0)
        sim.run(100.0)
        self.assertAlmostEqual(sim.get_current_time(), 100.0, 9)
        
    def test_get_time_step(self):
        self.assertEqual(sim.get_time_step(), 0.1)
        sim.setup(timestep=0.05)
        self.assertEqual(sim.get_time_step(), 0.05)
                
#===============================================================================
class SpikeSourceTest(unittest.TestCase):
    
    def test_end_of_poisson(self):
        sim.setup()
        # see ticket:127
        start = 100.0
        rate = 50.0
        duration = 400.0
        n = 10
        simtime = start + duration + 100.0
        PNs = sim.Population(n, sim.SpikeSourcePoisson, cellparams={'start': start, 'rate': rate, 'duration': duration})
        PNs.record()
        sim.run(simtime)
        spikes = PNs.getSpikes()[:,1]
        if sim.rank() == 0:
            assert min(spikes) >= start, min(spikes)
            assert max(spikes) <= start+duration, "%g > %g" % (max(spikes), start+duration)
            expected_count = duration/1000.0*rate*n
            diff = abs(len(spikes)-expected_count)/expected_count
            assert diff <= 0.1, "diff = %g. Expected count = %d, actual count = %d" % (diff, expected_count, len(spikes))
    
class FileTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        self.pop = sim.Population((3,3), sim.IF_curr_alpha)
        rng = random.NumpyRNG(123)
        v_reset  = -65.0
        v_thresh = -50.0
        uniformDistr = random.RandomDistribution(rng=rng, distribution='uniform', parameters=[v_reset, v_thresh])
        self.pop.randomInit(uniformDistr)
        self.cells_to_record = [self.pop[0,0], self.pop[1,1]]
        self.pop.record_v(self.cells_to_record)
        simtime = 10.0
        sim.run(simtime)
        
    def tearDown(self):
        if sim.rank() == 0:
            for ext in "txt", "pkl", "npz", "h5":
                filename = "temp_v.%s" % ext
                if os.path.exists(filename):
                    os.remove(filename)
    
    def test_text_file(self):
        output_file = recording.files.StandardTextFile("temp_v.txt", 'w')
        self.pop.print_v(output_file, gather=True, compatible_output=True)
        vm = self.pop.get_v()
        if sim.rank() == 0:
            self.assertEqual(vm.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 3))
            vm_fromfile = numpy.loadtxt(output_file.name)
            #print vm_fromfile
            self.assertEqual(vm_fromfile.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 2))
                
    def test_pickle_file(self):
        output_file = recording.files.PickleFile("temp_v.pkl", 'w')
        self.pop.print_v(output_file, gather=True, compatible_output=True)
        vm = self.pop.get_v()
        if sim.rank() == 0:
            self.assertEqual(vm.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 3))
            f = open(output_file.name)
            vm_fromfile, metadata = pickle.load(f)
            f.close()
            #print vm_fromfile
            self.assertEqual(vm_fromfile.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 2))
        
    def test_numpy_binary_file(self):
        output_file = recording.files.NumpyBinaryFile("temp_v.npz", 'w')
        self.pop.print_v(output_file, gather=True, compatible_output=True)
        vm = self.pop.get_v()
        if sim.rank() == 0:
            self.assertEqual(vm.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 3))
            vm_fromfile = numpy.load(output_file.name)['data']
            #print vm_fromfile
            self.assertEqual(vm_fromfile.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 2))
                
    def test_hdf5_array_file(self):
        output_file = recording.files.HDF5ArrayFile("temp_v.h5", 'w')
        self.pop.print_v(output_file, gather=True, compatible_output=True)
        vm = self.pop.get_v()
        if sim.rank() == 0:
            self.assertEqual(vm.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 3))
            #h5file = tables.openFile(output_file.name, 'r')
            h5file = recording.files.HDF5ArrayFile("temp_v.h5", 'r')
            vm_fromfile = h5file.read()
            h5file.close()
            self.assertEqual(vm_fromfile.shape, ((1+int(10.0/0.1))*len(self.cells_to_record), 2))
                
# ==============================================================================
if __name__ == "__main__":
    sim_name = utility.get_script_args(1)[0]
    
    sys.argv.remove(sim_name) # because unittest.main() processes sys.argv
    if sim_name == 'neuron':
        sys.argv = sys.argv[sys.argv.index('generictests.py'):]
    
    #print sys.argv
    sim = __import__("pyNN.%s" % sim_name, None, None, [sim_name])
    sim.setup()
    unittest.main()