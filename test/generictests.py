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
        self.cells = sim.create(sim.IF_curr_exp, n=2)
    
    def tearDown(self):
        pass
    
    def testSetGetParameters(self):
        """setParameters(), getParameters(): sanity check"""
        new_parameters = {}
        for name in sim.IF_curr_exp.default_parameters.keys():
            new_parameters[name] = numpy.random.uniform()
        self.cells[0].setParameters(**new_parameters)
        retrieved_parameters = self.cells[0].getParameters()
        self.assertEqual(new_parameters, retrieved_parameters)
        
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