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
            for cell in cell_list:
                for name in cell_class.default_parameters:
                    if name == 'spike_times':
                        i = [1.0, 2.0]
                        cell.__setattr__(name, i)
                        o = cell.__getattr__(name)
                        self.assertEqual(i, o)
                    else:
                        i = numpy.random.uniform(0.1, 100) # tau_refrac is always at least dt (=0.1)
                        cell.__setattr__(name, i)
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
            for cell in cell_list:
                new_parameters = {}
                for name in cell_class.default_parameters:
                    if name == 'spike_times':
                        new_parameters[name] = [1.0, 2.0]
                    else:
                        new_parameters[name] = numpy.random.uniform(0.1, 100) # tau_refrac is always at least dt (=0.1)
                cell.set_parameters(**new_parameters)
                retrieved_parameters = cell.get_parameters()
                self.assertEqual(set(new_parameters.keys()), set(retrieved_parameters.keys()))
                
                for name in new_parameters:
                    i = new_parameters[name]; o = retrieved_parameters[name]
                    if name != 'spike_times':
                        self.assertEqual(type(i), type(o), "%s: input: %s, output: %s" % (name, type(i), type(o)))
                        self.assertAlmostEqual(i, o,
                                               IDSetGetTest.decimal_places.get(name, IDSetGetTest.default_dp),
                                               "%s in %s: %s != %s" % (name, cell_class.__name__, i,o))
       
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
        err_msg = "%s != %s" % (self.input_spike_array, output_spike_array)
        self.assert_(arrays_almost_equal(self.input_spike_array, output_spike_array, 1e-13), err_msg)
    
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

class PopulationSetTest(unittest.TestCase):
    
    def setUp(self):
        sim.setup()
        cell_params = {
            'tau_m' : 20.,  'tau_syn_E' : 2.3,   'tau_syn_I': 4.5,
            'v_rest': -55., 'v_reset'   : -62.3, 'v_thresh' : -50.2,
            'cm'    : 1.,   'tau_refrac': 2.3}
        self.p1 = sim.Population((10,), sim.IF_curr_exp, cell_params)
        
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
                
# ==============================================================================
if __name__ == "__main__":
    simulator = sys.argv[1]
    sys.argv.remove(simulator) # because unittest.main() processes sys.argv
    sim = __import__("pyNN.%s" % simulator, None, None, [simulator])
    unittest.main()