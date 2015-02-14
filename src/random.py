"""
Provides wrappers for several random number generators (RNGs), giving them all a
common interface so that they can be used interchangeably in PyNN.

Classes:
    NumpyRNG           - uses the numpy.random.RandomState RNG
    GSLRNG             - uses the RNGs from the Gnu Scientific Library
    NativeRNG          - indicates to the simulator that it should use it's own,
                         built-in RNG
    RandomDistribution - produces random numbers from a specific distribution


:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
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


available_distributions = {
    'binomial':       ('n', 'p'),
    'gamma':          ('k', 'theta'),
    'exponential':    ('beta',),
    'lognormal':      ('mu', 'sigma'),
    'normal':         ('mu', 'sigma'),
    'normal_clipped': ('mu', 'sigma', 'low', 'high'),
    'normal_clipped_to_boundary':
                      ('mu', 'sigma', 'low', 'high'),
    'poisson':        ('lambda_',),
    'uniform':        ('low', 'high'),
    'uniform_int':    ('low', 'high'),
    'vonmises':       ('mu', 'kappa'),
}

MAX_REDRAWS = 1000  # for clipped distributions


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
            assert isinstance(seed, int), "`seed` must be an int, not a %s" % type(seed).__name__
        self.seed = seed
        # define some aliases
        self.random = self.next
        self.sample = self.next

    def __repr__(self):
        return "%s(seed=%r)" % (self.__class__.__name__, self.seed)

    def next(self, n=None, distribution=None, parameters=None, mask_local=None):
        """Return `n` random numbers from the specified distribution.

        If:
            * `n` is None, return a float,
            * `n` >= 1, return a Numpy array,
            * `n` < 0, raise an Exception,
            * `n` is 0, return an empty array.

        If called with distribution=None, returns uniformly distributed floats in the range [0, 1)
        """
        raise NotImplementedError


class WrappedRNG(AbstractRNG):

    def __init__(self, seed=None, parallel_safe=True):
        AbstractRNG.__init__(self, seed)
        self.parallel_safe = parallel_safe
        self.mpi_rank, self.num_processes = get_mpi_config()
        if self.seed is not None and not parallel_safe:
            self.seed += self.mpi_rank  # ensure different nodes get different sequences
            if self.mpi_rank != 0:
                logger.warning("Changing the seed to %s on node %d" % (self.seed, self.mpi_rank))

    def next(self, n=None, distribution=None, parameters=None, mask_local=None):
        if distribution is None:
            distribution = 'uniform'
            if parameters is None:
                parameters = (0.0, 1.0)
        if n == 0:
            rarr = numpy.random.rand(0)  # We return an empty array
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

    def _clipped(self, gen, low=-numpy.inf, high=numpy.inf, size=None):
        """ """
        res = gen(size)
        iterations = 0
        errmsg = "Maximum number of redraws exceeded. Check the parameterization of your distribution."
        if size is None:
            while res < low or res > high:
                # limit the number of iterations. Possibility of infinite loop, depending on parameters
                if iterations > MAX_REDRAWS:
                    raise Exception(errmsg)
                res = gen(size)
                iterations += 1
        else:
            idx = numpy.where((res > high) | (res < low))[0]
            while idx.size > 0:
                if iterations > MAX_REDRAWS:
                    raise Exception(errmsg)
                redrawn = gen(idx.size)
                res[idx] = redrawn
                idx = idx[numpy.where((redrawn > high) | (redrawn < low))[0]]
                iterations += 1
        return res

    def describe(self):
        return "%s() with seed %s for MPI rank %d (MPI processes %d). %s parallel safe." % (
            self.__class__.__name__, self.seed, self.mpi_rank, self.num_processes, self.parallel_safe and "Is" or "Not")


class NumpyRNG(WrappedRNG):
    """Wrapper for the :class:`numpy.random.RandomState` class (Mersenne Twister PRNG)."""
    translations = {
        'binomial':       ('binomial',     {'n': 'n', 'p': 'p'}),
        'gamma':          ('gamma',        {'k': 'shape', 'theta': 'scale'}),
        'exponential':    ('exponential',  {'beta': 'scale'}),
        'lognormal':      ('lognormal',    {'mu': 'mean', 'sigma': 'sigma'}),
        'normal':         ('normal',       {'mu': 'loc', 'sigma': 'scale'}),
        'normal_clipped': ('normal_clipped', {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'}),
        'normal_clipped_to_boundary':
                          ('normal_clipped_to_boundary', {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'}),
        'poisson':        ('poisson',      {'lambda_': 'lam'}),
        'uniform':        ('uniform',      {'low': 'low', 'high': 'high'}),
        'uniform_int':    ('randint',      {'low': 'low', 'high': 'high'}),
        'vonmises':       ('vonmises',     {'mu': 'mu', 'kappa': 'kappa'}),
    }

    def __init__(self, seed=None, parallel_safe=True):
        WrappedRNG.__init__(self, seed, parallel_safe)
        self.rng = numpy.random.RandomState()
        if self.seed is not None:
            self.rng.seed(self.seed)
        else:
            self.rng.seed()

    def __getattr__(self, name):
        """
        This is to give the PyNN RNGs the same methods as the wrapped RNGs
        (:class:`numpy.random.RandomState` or the GSL RNGs.)
        """
        return getattr(self.rng, name)

    def _next(self, distribution, n, parameters):
        # TODO: allow non-standardized distributions to pass through without translation
        distribution_np, parameter_map = self.translations[distribution]
        if set(parameters.keys()) != set(parameter_map.keys()):
            # all parameters must be provided. We do not provide default values (this can be discussed).
            errmsg = "Incorrect parameterization of random distribution. Expected %s, got %s."
            raise KeyError(errmsg % (parameter_map.keys(), parameters.keys()))
        parameters_np = dict((parameter_map[k], v) for k, v in parameters.items())
        if hasattr(self, distribution_np):
            f_distr = getattr(self, distribution_np)
        else:
            f_distr = getattr(self.rng, distribution_np)
        return f_distr(size=n, **parameters_np)

    def __deepcopy__(self, memo):
        obj = NumpyRNG.__new__(NumpyRNG)
        WrappedRNG.__init__(obj, seed=deepcopy(self.seed, memo),
                            parallel_safe=deepcopy(self.parallel_safe, memo))
        obj.rng = deepcopy(self.rng)
        return obj

    def normal_clipped(self, mu=0.0, sigma=1.0, low=-numpy.inf, high=numpy.inf, size=None):
        """ """
        # not sure how well this works with parallel_safe, mask_local
        gen = lambda n: self.rng.normal(loc=mu, scale=sigma, size=n)
        return self._clipped(gen, low=low, high=high, size=size)

    def normal_clipped_to_boundary(self, mu=0.0, sigma=1.0, low=-numpy.inf, high=numpy.inf, size=None):
        # Not recommended, used `normal_clipped` instead.
        # Provided because some models in the literature use this.
        res = self.rng.normal(loc=mu, scale=sigma, size=size)
        return numpy.maximum(numpy.minimum(res, high), low)


class GSLRNG(WrappedRNG):
    """Wrapper for the GSL random number generators."""
    translations = {
        'binomial':       ('binomial',       {'n': 'n', 'p': 'p'}),
        'gamma':          ('gamma',          {'k': 'k', 'theta': 'theta'}),
        'exponential':    ('exponential',    {'beta': 'mu'}),
        'lognormal':      ('lognormal',      {'mu': 'zeta', 'sigma': 'sigma'}),
        'normal':         ('normal',         {'mu': 'mu', 'sigma': 'sigma'}),
        'normal_clipped': ('normal_clipped', {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'}),
        'poisson':        ('poisson',        {'lambda_': 'mu'}),
        'uniform':        ('flat',           {'low': 'a', 'high': 'b'}),
        'uniform_int':    ('uniform_int',    {'low': 'low', 'high': 'high'}),
    }

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
        distribution_gsl, parameter_map = self.translations[distribution]
        if set(parameters.keys()) != set(parameter_map.keys()):
            # all parameters must be provided. We do not provide default values (this can be discussed).
            errmsg = "Incorrect parameterization of random distribution. Expected %s, got %s."
            raise KeyError(errmsg % (parameter_map.keys(), parameters.keys()))
        parameters_gsl = dict((parameter_map[k], v) for k, v in parameters.items())
        if hasattr(self, distribution_gsl):
            f_distr = getattr(self, distribution_gsl)
        else:
            f_distr = getattr(self.rng, distribution_gsl)
        # Has this been tested? If so, move most of _next to Wrapped RNG since there is almost complete overlap with NumpyRNG._next
        values = f_distr(size=n, **parameters_gsl)
        if n == 1:
            values = [values]  # to be consistent with NumpyRNG
        return values

    def uniform_int(self, low, high, size=None):
        return low + self.rng.uniform_int(high-low, size)

    def gamma(self, k, theta, size=None):
        """ """
        return self.rng.gamma(k, 1/theta, size)

    def normal(self, mu=0.0, sigma=1.0, size=None):
        """ """
        return mu + self.rng.gaussian(sigma, size)

    def normal_clipped(self, mu=0.0, sigma=1.0, low=-numpy.inf, high=numpy.inf, size=None):
        """ """
        gen = lambda n: self.normal(mu, sigma, n)
        return self._clipped(gen, low=low, high=high, size=size)

# should add a wrapper for the built-in Python random module.


class NativeRNG(AbstractRNG):
    """
    Signals that the simulator's own native RNG should be used.
    Each simulator module should implement a class of the same name which
    inherits from this and which sets the seed appropriately.
    """

    def __str__(self):
        return "NativeRNG(seed=%s)" % self.seed


class RandomDistribution(VectorizedIterable):
    """
    Class which defines a next(n) method which returns an array of `n` random
    numbers from a given distribution.

    Arguments:
        `distribution`:
            the name of a random number distribution.
        `parameters_pos`:
            parameters of the distribution, provided as a tuple. For the correct
            ordering, see `random.available_distributions`.
        `rng`:
            if present, should be a :class:`NumpyRNG`, :class:`GSLRNG` or
            :class:`NativeRNG` object.
        `parameters_named`:
            parameters of the distribution, provided as keyword arguments.

    Parameters may be provided either through `parameters_pos` or through
    `parameters_named`, but not both. All parameters must be provided, there
    are no default values. Parameter names are, in general, as used in Wikipedia.

    Examples:

    >>> rd = RandomDistribution('uniform', (-70, -50))
    >>> rd = RandomDistribution('normal', mu=0.5, sigma=0.1)
    >>> rng = NumpyRNG(seed=8658764)
    >>> rd = RandomDistribution('gamma', k=2.0, theta=5.0, rng=rng)

    Available distributions:

    ==========================  ====================  ====================================================
    Name                        Parameters            Comments
    --------------------------  --------------------  ----------------------------------------------------
    binomial                    n, p
    gamma                       k, theta
    exponential                 beta
    lognormal                   mu, sigma
    normal                      mu, sigma
    normal_clipped              mu, sigma, low, high  Values outside (low, high) are redrawn
    normal_clipped_to_boundary  mu, sigma, low, high  Values below/above low/high are set to low/high
    poisson                     lambda_               Trailing underscore since lambda is a Python keyword
    uniform                     low, high
    uniform_int                 low, high
    vonmises                    mu, kappa
    ==========================  ====================  ====================================================
    """

    def __init__(self, distribution, parameters_pos=None, rng=None, **parameters_named):
        """
        Create a new RandomDistribution.
        """
        self.name = distribution
        self.parameters = self._resolve_parameters(parameters_pos, parameters_named)
        if rng:
            assert isinstance(rng, AbstractRNG), "rng must be a pyNN.random RNG object"
            self.rng = rng
        else:  # use numpy.random.RandomState() by default
            self.rng = NumpyRNG()  # should we provide a seed?

    def next(self, n=None, mask_local=None):
        """Return `n` random numbers from the distribution."""
        res = self.rng.next(n=n,
                            distribution=self.name,
                            parameters=self.parameters,
                            mask_local=mask_local)
        return res

    def __str__(self):
        return "RandomDistribution('%(name)s', %(parameters)s, %(rng)s)" % self.__dict__

    def _resolve_parameters(self, positional, named):
        if positional is None:
            if set(named.keys()) != set(available_distributions[self.name]):
                errmsg = "Incorrect parameterization of random distribution. Expected %s, got %s."
                raise KeyError(errmsg % (available_distributions[self.name], tuple(named.keys())))
            return named
        elif len(named) == 0:
            expected_parameter_names = available_distributions[self.name]
            if len(positional) != len(expected_parameter_names):
                errmsg = "Incorrect number of parameters for random distribution. For %s received %s"
                raise ValueError(errmsg % (expected_parameter_names, positional))
            else:
                return dict((name, value) for name, value in zip(expected_parameter_names, positional))
        else:
            raise ValueError("Mixed positional and named parameters")
