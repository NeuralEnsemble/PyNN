"""
docstring missing
"""

import numpy as np
from neuron import h
from pyNN.random import NativeRNG, WrappedRNG

try:
    xrange
except NameError:
    xrange = range


class NativeRNG(NativeRNG, WrappedRNG):
    """
    Signals that the random numbers will be drawn by NEURON's Random() class and
    takes care of transforming pyNN parameters for the random distributions
    to NEURON parameters.
    """
    translations = {
        'binomial':       ('binomial',     ('n', 'p')),
        'gamma':          ('gamma',        ('k', 'theta')),
        'exponential':    ('negexp',       ('beta',)),
        'lognormal':      ('lognormal',    ('mu', 'sigma')),
        'normal':         ('normal',       ('mu', 'sigma')),
        'normal_clipped': ('normal_clipped', ('mu', 'sigma', 'low', 'high')),
        #'normal_clipped_to_boundary':
        #                  ('normal_clipped_to_boundary', {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'}),
        'poisson':        ('poisson',      ('lambda_',)),
        'uniform':        ('uniform',      ('low', 'high')),
        'uniform_int':    ('discunif',     ('low', 'high')),
        #'vonmises':       ('vonmises',     {'mu': 'mu', 'kappa': 'kappa'}),
    }

    def __init__(self, seed=None, parallel_safe=True):
        WrappedRNG.__init__(self, seed, parallel_safe)
        if self.seed is None:
            self.rng = h.Random()
        else:
            self.rng = h.Random(self.seed)
        self.last_distribution = None

    def _next(self, distribution, n, parameters):
        distribution_nrn, parameter_ordering = self.translations[distribution]
        if set(parameters.keys()) != set(parameter_ordering):
            # all parameters must be provided. We do not provide default values (this can be discussed).
            errmsg = "Incorrect parameterization of random distribution. Expected %s, got %s."
            raise KeyError(errmsg % (parameter_ordering, parameters.keys()))
        parameters_nrn = [parameters[k] for k in parameter_ordering]
        if hasattr(self, distribution_nrn):
            return getattr(self, distribution_nrn)(n, *parameters_nrn)
        else:
            return self._next_n(distribution_nrn, n, parameters_nrn)

    def _next_n(self, distribution_nrn, n, parameters_nrn):
        if self.last_distribution == distribution_nrn:
            values = np.fromiter((self.rng.repick() for i in xrange(n)), dtype=float, count=n)
        else:
            self.last_distribution = distribution_nrn
            f_distr = getattr(self.rng, distribution_nrn)
            values = np.empty((n,))
            values[0] = f_distr(*parameters_nrn)
            for i in xrange(1, n):
                values[i] = self.rng.repick()
        return values

    def gamma(self, n, k, theta):
        if k % 1 == 0:  # k is equal to an integer
            # Remap k, theta to mean, variance:
            #  gamma(k, 1/lambda) = erlang(k, lambda)
            #  mean(erlang) = k/lambda
            #  var(erlang) = k/lambda^2
            mean = k*theta
            variance = mean*theta
            return self._next_n("erlang", n, (mean, variance))
        else:
            raise Exception("The general case of the gamma distribution is not supported.")

    def normal(self, n, mu, sigma):
        return self._next_n("normal", n, (mu, sigma*sigma))

    def normal_clipped(self, n, mu, sigma, low, high):
        """ """
        gen = lambda n: self.normal(n, mu, sigma)
        return self._clipped(gen, low=low, high=high, size=n)

