"""
Unit tests for all simulators
$Id:$
"""

import sys
import unittest
import numpy
import os

def arrays_almost_equal(a, b, threshold):
    return (abs(a-b) < threshold).all()

# ==============================================================================
class IDSetGetTest(unittest.TestCase):
    """Tests of the ID.__setattr__()`, `ID.__getattr()` `ID.setParameters()`
    and `ID.getParameters()` methods."""
    
    def setUp(self):
        sim.setup()
        self.cells = {}
        self.populations = {}
        for cell_class in sim.list_standard_models():
            print "Creating ", cell_class
            self.cells[cell_class.__name__] = sim.create(cell_class, n=2)
            self.populations[cell_class.__name__] = sim.Population(2, cell_class)
    
    def tearDown(self):
        pass
    
    def testSetGet(self):
        """__setattr__(), __getattr__(): sanity check"""
        decimal_places = 6
        for cell_class in sim.list_standard_models():
            for cell in (self.cells[cell_class.__name__][0],
                         self.populations[cell_class.__name__][0]):
                print "Testing ", cell_class
                for name in cell_class.default_parameters:
                    i = numpy.random.uniform()
                    cell.__setattr__(name, i)
                    o = cell.__getattr__(name)
                    self.assertAlmostEqual(i, o, decimal_places, "%s: %s != %s" % (name, i,o))
    
    def testSetGetParameters(self):
        """setParameters(), getParameters(): sanity check"""
        # need to do for all cell types and for both single cells and cells in Populations
        # need to add similar test for native models in the sim-specific test files
        decimal_places = 6
        for cell in (self.cells['IF_curr_exp'][0], self.populations['IF_curr_exp'][0]):
            new_parameters = {}
            for name in sim.IF_curr_exp.default_parameters.keys():
                new_parameters[name] = numpy.random.uniform()
            cell.setParameters(**new_parameters)
            retrieved_parameters = cell.getParameters()
            self.assertEqual(new_parameters.keys(), retrieved_parameters.keys())
            
            for name in new_parameters:
                i = new_parameters[name]; o = retrieved_parameters[name]
                self.assertAlmostEqual(i, o, decimal_places, "%s: %s != %s" % (name,i,o))
        
class PopulationSpikesTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        self.spiketimes = numpy.arange(5,105,10.0)
        spiketimes_2D = self.spiketimes.reshape((len(self.spiketimes),1))
        self.input_spike_array = numpy.concatenate((numpy.zeros(spiketimes_2D.shape, 'float'), spiketimes_2D),
                                                   axis=1)
        self.p1 = sim.Population(1, sim.SpikeSourceArray, {'spike_times': self.spiketimes})
        self.p1.record()
        sim.run(100.0)
    
    def tearDown(self):
        pass
    
    def testGetSpikes(self):
        output_spike_array = self.p1.getSpikes()
        err_msg = "%s != %s" % (self.input_spike_array, output_spike_array)
        self.assert_(arrays_almost_equal(self.input_spike_array, output_spike_array, 1e-13), err_msg)
    
    

if __name__ == "__main__":
    simulator = sys.argv[1]
    sys.argv.remove(simulator) # because unittest.main() processes sys.argv
    sim = __import__("pyNN.%s" % simulator, None, None, [simulator])
    unittest.main()