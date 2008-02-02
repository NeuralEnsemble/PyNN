"""
Unit tests for all simulators
$Id:$
"""

import sys
import pyNN.nest2 as sim
import pyNN.common as common
import pyNN.random as random
import unittest
import numpy
import os

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
        
        

if __name__ == "__main__":
    unittest.main()