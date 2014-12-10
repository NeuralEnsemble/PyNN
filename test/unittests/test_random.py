"""
Unit tests for pyNN/random.py.
"""

import pyNN.random as random
import numpy
try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from neuron import h
except ImportError:
    have_nrn = False
else:
    have_nrn = True
    from pyNN.neuron.random import NativeRNG


def assert_arrays_almost_equal(a, b, threshold):
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        raise unittest.TestCase.failureException(err_msg)


# ==============================================================================
class SimpleTests(unittest.TestCase):
    """Simple tests on a single RNG function."""

    def setUp(self):
        self.rnglist = [random.NumpyRNG(seed=987)]
        for rng in self.rnglist:
            rng.mpi_rank=0; rng.num_processes=1
        if random.have_gsl:
            self.rnglist.append(random.GSLRNG(seed=654))
        if have_nrn:
            self.rnglist.append(NativeRNG(seed=321))

    def testNextNone(self):
        """Calling next() with no number argument should return a float."""
        for rng in self.rnglist:
            self.assertIsInstance(rng.next(distribution='uniform', parameters={'low': 0, 'high': 1}), float)

    def testNextOne(self):
        """Calling next() with n=1 should return an array."""
        for rng in self.rnglist:
            self.assertIsInstance(rng.next(1, 'uniform', {'low': 0, 'high': 1}), numpy.ndarray)
            self.assertIsInstance(rng.next(n=1, distribution='uniform', parameters={'low': 0, 'high': 1}), numpy.ndarray)
            self.assertEqual(rng.next(1, distribution='uniform', parameters={'low': 0, 'high': 1}).shape, (1,))

    def testNextTwoPlus(self):
        """Calling next(n=m) where m > 1 should return an array."""
        for rng in self.rnglist:
            self.assertEqual(len(rng.next(5, 'uniform', {'low': 0, 'high': 1})), 5)

    def testNonPositiveN(self):
        """Calling next(m) where m < 0 should raise a ValueError."""
        for rng in self.rnglist:
            self.assertRaises(ValueError, rng.next, -1, 'uniform', {'low': 0, 'high': 1})

    def testNZero(self):
        """Calling next(0) should return an empty array."""
        for rng in self.rnglist:
            self.assertEqual(len(rng.next(0)), 0)

    def test_invalid_seed(self):
        self.assertRaises(AssertionError, random.NumpyRNG, seed=2.3)


class ParallelTests(unittest.TestCase):

    def setUp(self):
        self.rng_types = [random.NumpyRNG]
        if random.have_gsl:
            self.rng_types.append(random.GSLRNG)
        if have_nrn:
            self.rng_types.append(NativeRNG)
        self.orig_mpi_config = random.get_mpi_config

    def tearDown(self):
        random.get_mpi_config = self.orig_mpi_config

    def test_parallel_unsafe(self):
        for rng_type in self.rng_types:
            random.get_mpi_config = lambda: (0, 2)
            rng0 = rng_type(seed=1000, parallel_safe=False)
            random.get_mpi_config = lambda: (1, 2)
            rng1 = rng_type(seed=1000, parallel_safe=False)
            self.assertEqual(rng0.seed, 1000)
            self.assertEqual(rng1.seed, 1001)
            draw0 = rng0.next(5, 'uniform', {'low': 0, 'high': 1},)
            draw1 = rng1.next(5, 'uniform', {'low': 0, 'high': 1},)
            self.assertEqual(len(draw0), 5//2+1)
            self.assertEqual(len(draw1), 5//2+1)
            self.assertNotEqual(draw0.tolist(), draw1.tolist())

    def test_parallel_safe_with_mask_local(self):
        for rng_type in self.rng_types:
            random.get_mpi_config = lambda: (0, 2)
            rng0 = rng_type(seed=1000, parallel_safe=True)
            random.get_mpi_config = lambda: (1, 2)
            rng1 = rng_type(seed=1000, parallel_safe=True)
            draw0 = rng0.next(5, 'uniform', {'low': 0, 'high': 1}, mask_local=numpy.array((1,0,1,0,1), bool))
            draw1 = rng1.next(5, 'uniform', {'low': 0, 'high': 1}, mask_local=numpy.array((0,1,0,1,0), bool))
            self.assertEqual(len(draw0), 3)
            self.assertEqual(len(draw1), 2)
            self.assertNotEqual(draw0.tolist(), draw1.tolist())

    def test_parallel_safe_with_mask_local_False(self):
        for rng_type in self.rng_types:
            random.get_mpi_config = lambda: (0, 2)
            rng0 = rng_type(seed=1000, parallel_safe=True)
            random.get_mpi_config = lambda: (1, 2)
            rng1 = rng_type(seed=1000, parallel_safe=True)
            draw0 = rng0.next(5, 'uniform', {'low': 0, 'high': 1}, mask_local=False)
            draw1 = rng1.next(5, 'uniform', {'low': 0, 'high': 1}, mask_local=False)
            self.assertEqual(len(draw0), 5)
            self.assertEqual(len(draw1), 5)
            self.assertEqual(draw0.tolist(), draw1.tolist())

    def test_permutation(self):
        # only works for NumpyRNG at the moment. pygsl has a permutation module, but I can't find documentation for it.
        random.get_mpi_config = lambda: (0, 2)
        rng0 = random.NumpyRNG(seed=1000, parallel_safe=True)
        random.get_mpi_config = lambda: (1, 2)
        rng1 = random.NumpyRNG(seed=1000, parallel_safe=True)
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
        random.get_mpi_config = lambda: (0, 1)
        self.rnglist = [random.NumpyRNG(seed=987)]
        if random.have_gsl:
            self.rnglist.append(random.GSLRNG(seed=654))
        if have_nrn:
            self.rnglist.append(NativeRNG(seed=321))

    def test_uniform(self):
        rd = random.RandomDistribution(distribution='uniform', low=-1.0, high=3.0, rng=self.rnglist[0])
        vals = rd.next(100)
        assert vals.min() >= -1.0
        assert vals.max() < 3.0
        assert abs(vals.mean() - 1.0) < 0.2

    def test_gaussian(self):
        mean = 1.0
        std = 1.0
        rd1 = random.RandomDistribution('normal', mu=mean, sigma=std, rng=self.rnglist[0])
        vals_list = [rd1.next(100)]
        for vals in vals_list:
            assert vals.min() > mean-4*std
            assert vals.min() < mean+4*std
            assert abs(vals.mean() - mean) < 0.2, abs(vals.mean() - mean)

    def test_gamma(self):
        a = 2.0
        b = 0.5
        for rng in self.rnglist:
            rd = random.RandomDistribution('gamma', k=a, theta=1/b, rng=rng)
            vals = rd.next(100)
            # need to check vals are as expected
            str(rd)  # should be in a separate test

    def test_boundaries(self):
        rd = random.RandomDistribution('normal_clipped_to_boundary',
                                       mu=0, sigma=1, low=-0.5, high=0.5,
                                       rng=self.rnglist[0])
        vals = rd.next(1000)
        assert vals.min() == -0.5
        assert vals.max() == 0.5
        assert abs(vals.mean()) < 0.05, vals.mean()
        rd = random.RandomDistribution(distribution='normal_clipped',
                                       mu=0, sigma=1, low=0, high=1,
                                       rng=self.rnglist[0])
        vals = rd.next(1000)
        assert vals.min() >= 0
        assert vals.max() < 1.0

    def test_positional_args(self):
        for rng in self.rnglist:
            rd1 = random.RandomDistribution('normal', (0.5, 0.2), rng)
            self.assertEqual(rd1.parameters, {'mu': 0.5, 'sigma': 0.2})
            self.assertEqual(rd1.rng, rng)
        self.assertRaises(ValueError, random.RandomDistribution, 'normal', (0.5,))
        self.assertRaises(ValueError, random.RandomDistribution, 'normal', (0.5, 0.2), mu=0.5, sigma=0.2)

    def test_max_redraws(self):
        # for certain parameterizations, clipped distributions can require a very large, possibly infinite
        # number of redraws. This should be caught.
        for rng in self.rnglist:
            rd1 = random.RandomDistribution('normal_clipped', mu=0, sigma=1, low=5, high=numpy.inf, rng=rng)
            self.assertRaises(Exception, rd1.next, 1000)


# ==============================================================================
if __name__ == "__main__":
    unittest.main()
