"""
Unit tests for pyNN/random.py. 
$Id$
"""

import pyNN.random as random
import numpy
import unittest

def assert_arrays_almost_equal(a, b, threshold):
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        raise unittest.TestCase.failureException(err_msg)

# ==============================================================================
class SimpleTests(unittest.TestCase):
    """Simple tests on a single RNG function."""
    
    def setUp(self):
        self.rnglist = [random.NumpyRNG(seed=987),random.GSLRNG(seed=654)]
    
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

    def test_invalid_seed(self):
        self.assertRaises(AssertionError, random.NumpyRNG, seed=2.3)

class ParallelTests(unittest.TestCase):

    def test_parallel_unsafe(self):
        rng0 = random.NumpyRNG(seed=1000, rank=0, num_processes=2, parallel_safe=False)
        rng1 = random.NumpyRNG(seed=1000, rank=1, num_processes=2, parallel_safe=False)
        draw0 = rng0.next(5)
        draw1 = rng1.next(5)
        self.assertEqual(len(draw0), 5/2+1)
        self.assertEqual(len(draw1), 5/2+1)
        self.assertNotEqual(draw0.tolist(), draw1.tolist())

    def test_parallel_safe(self):
        rng0 = random.NumpyRNG(seed=1000, rank=0, num_processes=2, parallel_safe=True)
        rng1 = random.NumpyRNG(seed=1000, rank=1, num_processes=2, parallel_safe=True)
        draw0 = rng0.next(5)
        draw1 = rng1.next(5)
        self.assertEqual(len(draw0), 3)
        self.assertEqual(len(draw1), 2)
        self.assertNotEqual(draw0.tolist(), draw1.tolist())

    def test_permutation(self):
        rng0 = random.NumpyRNG(seed=1000, rank=0, num_processes=2, parallel_safe=True)
        rng1 = random.NumpyRNG(seed=1000, rank=1, num_processes=2, parallel_safe=True)
        A = range(10)
        perm0 = rng0.permutation(A)
        perm1 = rng1.permutation(A)
        assert_arrays_almost_equal(perm0, perm1, 1e-99)

class NativeRNGTests(unittest.TestCase):
    
    def test_create(self):
        rng = random.NativeRNG(seed=8274528)
        str(rng)

class RandomDistributionTests(unittest.TestCase):
    
    def setUp(self):
        self.rnglist = [random.NumpyRNG(seed=9876),random.GSLRNG(seed=6543)]
        
    def test_uniform(self):
        rd = random.RandomDistribution(distribution='uniform', parameters=[-1.0, 3.0], rng=self.rnglist[0]) 
        vals = rd.next(100)
        assert vals.min() >= -1.0
        assert vals.max() < 3.0
        assert abs(vals.mean() - 1.0) < 0.2
        # GSL uniform is always between 0 and 1
        rd = random.RandomDistribution(distribution='uniform', parameters=[], rng=self.rnglist[1])
        vals = rd.next(100)
        assert vals.min() >= 0.0
        assert vals.max() < 1.0
        assert abs(vals.mean() - 0.5) < 0.05
        
        
    def test_gaussian(self):
        mean = 1.0
        std = 1.0
        rd1 = random.RandomDistribution(distribution='normal', parameters=[mean, std], rng=self.rnglist[0])
        vals1 = rd1.next(100)
        # GSL gaussian is always centred on zero
        rd2 = random.RandomDistribution(distribution='gaussian', parameters=[std], rng=self.rnglist[1])
        vals2 = mean + rd2.next(100)
        for vals in vals1, vals2:
            assert vals.min() > mean-4*std
            assert vals.min() < mean+4*std
            assert abs(vals.mean() - mean) < 0.2, abs(vals.mean() - mean)
            
    def test_gamma(self):
        a = 0.5
        b = 0.5
        for rng in self.rnglist:
            rd = random.RandomDistribution(distribution='gamma', parameters=[a, b], rng=rng)
            vals = rd.next(100)
            # need to check vals are as expected
            str(rd) # should be in a separate test
            
    def test_boundaries(self):
        rd = random.RandomDistribution(distribution='uniform', parameters=[-1.0, 1.0],
                                       rng=self.rnglist[0], boundaries=[0.0, 1.0],
                                       constrain="clip")
        vals = rd.next(100)
        assert vals.min() == 0
        assert vals.max() < 1.0
        assert abs(vals.mean() - 0.25) < 0.05
        rd = random.RandomDistribution(distribution='uniform', parameters=[-1.0, 1.0],
                                       rng=self.rnglist[0], boundaries=[0.0, 1.0],
                                       constrain="redraw")
        vals = rd.next(100)
        assert vals.min() >= 0
        assert vals.max() < 1.0
        assert abs(vals.mean() - 0.5) < 0.05
        val = rd.next()
        rd = random.RandomDistribution(distribution='uniform', parameters=[-1.0, 1.0],
                                       rng=self.rnglist[0], boundaries=[0.0, 1.0],
                                       constrain=None)
        self.assertRaises(Exception, rd.next)

# ==============================================================================            
if __name__ == "__main__":
    unittest.main()      
            
            
            
            
            
