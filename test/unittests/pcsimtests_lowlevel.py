"""
Unit tests for pyNN.pcsim package

    Unit tests for verifying the correctness of the low level function based 
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
from numpy import arange

# ==============================================================================
class CreationTest(unittest.TestCase):
    """Tests of the create() function."""
    
    def setUp(self):
        setup()        
    
    def tearDown(self):
        end()
    
    def testCreateStandardCell(self):
        
        cellid = create(IF_curr_alpha)
        assert isinstance(simulator.net.object(cellid), LIFCurrAlphaNeuron)
        
    def testCreateStandardCells(self):
                
        """create(): Creating multiple cells should return a list of pcsim SimObject IDs"""
        cell_ids = create(IF_curr_alpha, n = 10)
        self.assertEqual( len(cell_ids), 10 )
        for i in cell_ids:
            assert isinstance(simulator.net.object(i), LIFCurrAlphaNeuron)        
       
    def testCreateStandardCellsWithNegative_n(self):
        """create(): n must be positive definite"""        
        self.assertRaises(AssertionError, create, IF_curr_alpha, n=-1)
       
    def testCreateStandardCellWithParams(self):
        """create(): Parameters set on creation should be the same as retrieved with pypcsim"""        
        
        cellid = create(IF_curr_alpha,{'tau_syn_E':3.141592654}) # ms
        self.assertAlmostEqual(simulator.net.object(cellid).TauSynExc, 0.003141592654, places = 6) #s
    
    def testCreatePCSIMCell(self):
        """create(): First cell created should have index 0."""        
        
        cellid = create(CbLifNeuron, {'Rm': 2.3e6,'Cm': 1e-9})
        assert isinstance(simulator.net.object(cellid), CbLifNeuron )
        self.assertAlmostEqual(simulator.net.object(cellid).Rm, 2.3e6, places = 6)
        self.assertAlmostEqual(simulator.net.object(cellid).Cm,  1e-9, places = 6)
    
    def testCreateNonStandardCell(self):
         """create(): Trying to create a cell type which is not a method of StandardCells should raise an AttributeError."""
         self.assertRaises(common.InvalidModelError, create, 'qwerty')
    
    def testCreateWithInvalidParameter(self):
        """create(): Creating a cell with an invalid parameter should raise an Exception."""
        self.assertRaises(common.NonExistentParameterError, create, IF_curr_alpha, {'tau_foo':3.141592654})


# ==============================================================================
class ConnectionTest(unittest.TestCase):
    """Tests of the connect() function."""
    
    def setUp(self):
        setup(max_delay=5.0)
        self.postcells = create(IF_curr_alpha,n=3)
        self.precells = create(SpikeSourcePoisson,n=5)
        
    def tearDown(self):
        end()
        
    def testConnectTwoCells(self):
        """connect(): The first connection created should have id 0."""        
        
        conn = connect(self.precells[0],self.postcells[0])
        assert isinstance( conn[0].pcsim_connection, SimpleScalingSpikingSynapse)
        
    def testConnectTwoCellsWithWeight(self):
        """connect(): Weight set should match weight retrieved."""        
        
        conn_id = connect(self.precells[0],self.postcells[0],weight=0.1234)[0]
        weight = conn_id.pcsim_connection.W
        self.assertAlmostEqual( weight*1e9 , 0.1234 , places = 8, msg = "Weight set (0.1234) does not match weight retrieved (%s)" % (weight*1e9,) )
    
    def testConnectTwoCellsWithDelay(self):
        """connect(): Delay set should match delay retrieved."""
        
        conn_id = connect(self.precells[0],self.postcells[0],delay=4.321)[0]
        delay = conn_id.pcsim_connection.delay
        self.assertAlmostEqual( delay, 0.004321, places = 8, msg = "Delay set (0.004321 s) does not match delay retrieved (%s s)." % delay )
        
    
    def testConnectManyToOne(self):
        """connect(): Connecting n sources to one target should return a list of size n, each element being the id number of a netcon."""
        conn_ids = connect(self.precells,self.postcells[0])
        self.assertEqual( len( conn_ids ), len(self.precells) )
        
    def testConnectOneToMany(self):
        """connect(): Connecting one source to n targets should return a list of target ports."""
        conn_ids = connect(self.precells[0],self.postcells)
        self.assertEqual( len( conn_ids ), len(self.postcells) )
        
    def testConnectManyToMany(self):
        """connect(): Connecting m sources to n targets should return a list of length m x n"""
        conn_ids = connect(self.precells,self.postcells)
        self.assertEqual(len(conn_ids), len(self.postcells)*len(self.precells))
        
    def testConnectWithProbability(self):
        """connect(): If p=0.5, it is very unlikely that either zero or the maximum number of connections should be created."""
        connlist = connect(self.precells,self.postcells,p=0.5)
        assert 0 < len(connlist) < len(self.precells)*len(self.postcells), 'Number of connections is %d: this is very unlikely (although possible).' % len(connlist)
    
    def testConnectNonExistentPreCell(self):
        """connect(): Connecting from non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, connect, SimObject.ID(0,0,15,234).packed(), self.postcells[0])
        
    def testConnectNonExistentPostCell(self):
        """connect(): Connecting to a non-existent cell should raise a ConnectionError."""
        self.assertRaises(common.ConnectionError, connect, self.precells[0], SimObject.ID(0,0,15,2343).packed() )
    
    def testInvalidSourceId(self):
        """connect(): sources must be integers."""
        self.precells.append('74367598')
        self.assertRaises(common.ConnectionError, connect, self.precells, self.postcells)
    
    def testInvalidTargetId(self):
        """connect(): targets must be integers."""
        self.postcells.append([])
        self.assertRaises(common.ConnectionError, connect, self.precells, self.postcells)

# ==============================================================================
class SetValueTest(unittest.TestCase):
    
    def setUp(self):
        setup()
        self.cells = create(IF_curr_exp, n = 10)    
        
    def testSetFloat(self):
                
        set(self.cells, 'tau_m', 35.7)
        for cell in self.cells:
            self.assertAlmostEqual( simulator.net.object(cell).taum*1000.0 , 35.7, places = 5 )
            
    #def testSetString(self):
    #    set(self.cells,IF_curr_exp,'param_name','string_value')
    ## note we don't currently have any models with string parameters, so
    ## this is all commented out
    #    for cell in self.cells:
    #        assert HocToPy.get('cell%d.param_name' % cell, 'string') == 'string_value'

    def testSetDict(self):
        
        set(self.cells,{'tau_m':35.7,'tau_syn_E':5.432})
        for cell in self.cells:
            self.assertAlmostEqual( simulator.net.object(cell).taum*1000.0, 35.7, places = 5     )
            self.assertAlmostEqual( simulator.net.object(cell).TauSynExc*1000.0, 5.432, places = 6 )

    def testSetNonExistentParameter(self):
        # note that although syn_shape is added to the parameter dict when creating
        # an IF_curr_exp, it is not a valid parameter to be changed later.
        self.assertRaises(common.NonExistentParameterError, set, self.cells, 'some_param','some_value')

# ==============================================================================
class RecordTest(unittest.TestCase):
    
    def setUp(self):
        setup()
        
    def tearDown(self):
        end()
  
    
    def testRecordSpikesTextFormat(self):
        setup()
        spiking_nrn = create(SpikingInputNeuron, {}, 10)
        for i,n in enumerate(spiking_nrn):
            simulator.net.object(n).setSpikes( [ (i+1) * 0.001 + t for t in arange(0,1,0.01) ] )                    
        record( spiking_nrn, "recordTestSpikeFile1.txt")                
        run(1000)
	end(compatible_output=False) 
        # Now check the contents of the file
        f = file('recordTestSpikeFile1.txt', 'r')
        expected_id = 0;
        expected_spike_time = 0.001;
        for line in f:
            if len(line.split()) > 0 and line.split()[0][0] != '#':
                id, spike_time = line.split()
                self.assertEqual( expected_id, int(id))
                self.assertAlmostEqual( expected_spike_time, float(spike_time), places = 7 )
                expected_id = (expected_id + 1)  % 10
                expected_spike_time += 0.001        
                
    def testRecordVmTextFormat(self):        
        analog_nrns = create(AnalogInputNeuron, {}, 10)
        for i, n in enumerate(analog_nrns):
            simulator.net.object(n).setAnalogValues( [ (k + i) % 10 for k in xrange(1000)])
        record_v(analog_nrns, "recordTestVmFile1.txt")
        run(100)
        end(compatible_output=False)
        f = file('recordTestVmFile1.txt', 'r')
        expected_id = 0
        for line in f:
            values = line.split()
            values = [ float(i) for i in values ]
            self.assertEqual( expected_id, int(values[0]) )            
            self.assertEqual( [ float((expected_id + k) % 10) for k in xrange(1000)], values[1:] )
            expected_id += 1
        
        
        
        
    pass # to do later

if __name__ == "__main__":
    unittest.main()
    
