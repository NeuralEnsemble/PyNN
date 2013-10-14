"""
Provides wrappers for several random number generators (RNGs), giving them all a
common interface so that they can be used interchangeably in PyNN.

Note however that we have so far made no effort to implement parameter
translation, and parameter names/order may be different for the different RNGs.

Classes:
    NumpyRNG           - uses the numpy.random.RandomState RNG
    GSLRNG             - uses the RNGs from the Gnu Scientific Library
    NativeRNG          - indicates to the simulator that it should use it's own,
                         built-in RNG
    RandomDistribution - produces random numbers from a specific distribution


:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import sys
from copy import deepcopy
import logging
import numpy.random
from lazyarray import VectorizedIterable

try:
    import pygsl.rng
    have_gsl = True
except (ImportError, Warning):
    have_gsl = False
import time

logger = logging.getLogger("PyNN")


def get_mpi_config():
    try:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.rank
        num_processes = MPI.COMM_WORLD.size
    except ImportError:
        mpi_rank = 0
        num_processes = 1
    return mpi_rank, num_processes


class AbstractRNG(object):
    """Abstract class for wrapping random number generators. The idea is to be
    able to use either simulator-native rngs, which may be more efficient, or a
    standard Python rng, e.g. a numpy.random.RandomState object, which would
    allow the same random numbers to be used across different simulators, or
    simply to read externally-generated numbers from files."""

    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int), "`seed` must be an int, not a %s" % (type(seed).__name__,)
        self.seed = seed
        # define some aliases
        self.random = self.next
        self.sample = self.next

    def __repr__(self):
        return "%s(seed=%r)" % (self.__class__.__name__, self.seed)

    def next(self, n=None, distribution='uniform', parameters=[], mask_local=None):
        """Return `n` random numbers from the specified distribution.

        If:
            * `n` is None, return a float,
            * `n` >= 1, return a Numpy array,
            * `n` < 0, raise an Exception,
            * `n` is 0, return an empty array.
            """
        # arguably, rng.next() should return a float, rng.next(1) an array of length 1
        raise NotImplementedError


class WrappedRNG(AbstractRNG):

    def __init__(self, seed=None, parallel_safe=True):
        AbstractRNG.__init__(self, seed)
        self.parallel_safe = parallel_safe
        self.mpi_rank, self.num_processes = get_mpi_config()
        if self.seed is not None and not parallel_safe:
            self.seed += self.mpi_rank # ensure different nodes get different sequences
            if self.mpi_rank != 0:
                logger.warning("Changing the seed to %s on node %d" % (self.seed, self.mpi_rank))

    def next(self, n=None, distribution='uniform', parameters=[], mask_local=None):
        if n == 0:
            rarr = numpy.random.rand(0) # We return an empty array
        elif n is None:
            rarr = self._next(distribution, 1, parameters)
        elif n > 0:
            if self.num_processes > 1 and not self.parallel_safe:
                # n is the number for the whole model, so if we do not care about
                # having exactly the same random numbers independent of the
                # number of processors (m), we only need generate n/m+1 per node
                # (assuming round-robin distribution of cells between processors)
                if mask_local is None:
                    n = n/self.num_processes + 1
                elif mask_local is not False:
                    n = mask_local.sum()
            rarr = self._next(distribution, n, parameters)
        else:
            raise ValueError("The sample number must be positive")
        if not isinstance(rarr, numpy.ndarray):
            rarr = numpy.array(rarr)
        if self.parallel_safe and self.num_processes > 1:
            if hasattr(mask_local, 'size'):      # strip out the random numbers that
                assert mask_local.size == n      # should be used on other processors.
                rarr = rarr[mask_local]
        if n is None:
            return rarr[0]
        else:
            return rarr
    next.__doc__ = AbstractRNG.next.__doc__

    def __getattr__(self, name):
        """
        This is to give the PyNN RNGs the same methods as the wrapped RNGs
        (:class:`numpy.random.RandomState` or the GSL RNGs.)
        """
        return getattr(self.rng, name)

    def describe(self):
        return "%s() with seed %s for MPI rank %d (MPI processes %d). %s parallel safe." % (
            self.__class__.__name__, self.seed, self.mpi_rank, self.num_processes, self.parallel_safe and "Is" or "Not")


class NumpyRNG(WrappedRNG):
    """Wrapper for the :class:`numpy.random.RandomState` class (Mersenne Twister PRNG)."""

    def __init__(self, seed=None, parallel_safe=True):
        WrappedRNG.__init__(self, seed, parallel_safe)
        self.rng = numpy.random.RandomState()
        if self.seed is not None:
            self.rng.seed(self.seed)
        else:
            self.rng.seed()

    def _next(self, distribution, n, parameters):
        return getattr(self.rng, distribution)(size=n, *parameters)

    def __deepcopy__(self, memo):
        obj = NumpyRNG.__new__(NumpyRNG)
        WrappedRNG.__init__(obj, seed=deepcopy(self.seed, memo),
                             parallel_safe=deepcopy(self.parallel_safe, memo))
        obj.rng = deepcopy(self.rng)
        return obj


class GSLRNG(WrappedRNG):
    """Wrapper for the GSL random number generators."""

    def __init__(self, seed=None, type='mt19937', parallel_safe=True):
        if not have_gsl:
            raise ImportError("GSLRNG: Cannot import pygsl")
        WrappedRNG.__init__(self, seed, parallel_safe)
        self.rng = getattr(pygsl.rng, type)()
        if self.seed is not None:
            self.rng.set(self.seed)
        else:
            self.seed = int(time.time())
            self.rng.set(self.seed)

    def __getattr__(self, name):
        """This is to give GSLRNG the same methods as the GSL RNGs."""
        return getattr(self.rng, name)

    def _next(self, distribution, n, parameters):
        p = parameters + [n]
        values = getattr(self.rng, distribution)(*p)
        if n == 1:
            values = [values]  # to be consistent with NumpyRNG
        return values


# should add a wrapper for the built-in Python random module.


class NativeRNG(AbstractRNG):
    """
    Signals that the simulator's own native RNG should be used.
    Each simulator module should implement a class of the same name which
    inherits from this and which sets the seed appropriately.
    """

    def __str__(self):
        return "AbstractRNG(seed=%s)" % self.seed


class RandomDistribution(VectorizedIterable):
    """
    Class which defines a next(n) method which returns an array of `n` random
    numbers from a given distribution.

    Arguments:
        `rng`:
            if present, should be a :class:`NumpyRNG` or :class:`GSLRNG` object.
        `distribution`:
            the name of a method supported by the underlying random number
            generator object.
        `parameters`:
            a list or tuple containing the arguments expected by the underlying
            method in the correct order. Named arguments are not yet supported.
        `boundaries`:
            a tuple ``(min, max)`` used to specify explicitly, for distributions
            like Gaussian, gamma or others, hard boundaries for the parameters.
            If parameters are drawn outside those boundaries, the policy applied
            will depend on the `constrain` parameter.
        `constrain`:
            controls the policy for weights out of the specified boundaries.
            If "clip", random numbers are clipped to the boundaries.
            If "redraw", random numbers are drawn till they fall within the boundaries.

    Note that NumpyRNG and GSLRNG distributions may not have the same names,
    e.g., 'normal' for NumpyRNG and 'gaussian' for GSLRNG, and the arguments
    may also differ.
    """

    def __init__(self, distribution='uniform', parameters=[], rng=None,
                 boundaries=None, constrain="clip"):
        """
        Create a new RandomDistribution.
        """
        self.name = distribution
        assert isinstance(parameters, (list, tuple, dict)), "The parameters argument must be a list or tuple or dict"
        self.parameters = parameters
        self.boundaries = boundaries
        if self.boundaries:
            self.min_bound = min(self.boundaries)
            self.max_bound = max(self.boundaries)
        self.constrain  = constrain
        if rng:
            assert isinstance(rng, AbstractRNG), "rng must be a pyNN.random RNG object"
            self.rng = rng
        else: # use numpy.random.RandomState() by default
            self.rng = NumpyRNG()

    def next(self, n=None, mask_local=None):
        """Return `n` random numbers from the distribution."""
        res = self.rng.next(n=n,
                            distribution=self.name,
                            parameters=self.parameters,
                            mask_local=mask_local)
        if self.boundaries:
            if isinstance(res, numpy.float): 
                res = numpy.array([res])
            if self.constrain == "clip":
                return numpy.maximum(numpy.minimum(res, self.max_bound), self.min_bound)
            elif self.constrain == "redraw": # not sure how well this works with parallel_safe, mask_local
                if len(res) == 1:
                    while not ((res > self.min_bound) and (res < self.max_bound)):
                        res = self.rng.next(n=n, distribution=self.name, parameters=self.parameters, mask_local=mask_local)
                    return res
                else:
                    idx = numpy.where((res > self.max_bound) | (res < self.min_bound))[0]
                    while len(idx) > 0:
                        res[idx] = self.rng.next(n=n, distribution=self.name, parameters=self.parameters, mask_local=mask_local)[idx]
                        idx = numpy.where((res > self.max_bound) | (res < self.min_bound))[0]
                    return res
            else:
                raise Exception("This constrain method (%s) does not exist" %self.constrain)
        return res

    def __str__(self):
        return "RandomDistribution('%(name)s', %(parameters)s, %(rng)s)" % self.__dict__
