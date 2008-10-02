"""
Unit tests for pyNN/random.py. 
$Id$
"""

import pyNN.random as random
import numpy
import unittest

# ==============================================================================
class SimpleTests(unittest.TestCase):
    """Simple tests on a single RNG function."""
    
    def setUp(self):
        self.rnglist = [random.NumpyRNG(),random.GSLRNG()]
    
    def testNextOne(self):
        """Calling next() with no arguments or with n=1 should return a float."""
        for rng in self.rnglist:
            assert isinstance(rng.next(),float)
            assert isinstance(rng.next(1),float)
            assert isinstance(rng.next(n=1),float)
    
    def testNextTwoPlus(self):
        """Calling next(n=m) where m > 1 should return an array."""
        for rng in self.rnglist:
            self.assertEqual(len(rng.next(5)),5)
            self.assertEqual(len(rng.next(n=5)),5)
            
    def testNonPositiveN(self):
        """Calling next(m) where m < 0 should raise a ValueError."""
        for rng in self.rnglist:
            self.assertRaises(ValueError,rng.next,-1)

    def testNZero(self):
        """Calling next(0) should return an empty array."""
        for rng in self.rnglist:
	    self.assertEqual(len(rng.next(0)), 0)

# ==============================================================================            
if __name__ == "__main__":
    unittest.main()      
            
            
            
            
            
