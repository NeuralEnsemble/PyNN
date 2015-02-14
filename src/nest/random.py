"""
docstring missing
"""

from numbers import Real
from copy import copy
from pyNN.random import NativeRNG

NEST_RDEV_TYPES = ['binomial', 'binomial_clipped', 'binomial_clipped_to_boundary',
                   'exponential', 'exponential_clipped', 'exponential_clipped_to_boundary',
                   'gamma', 'gamma_clipped', 'gamma_clipped_to_boundary', 'gsl_binomial',
                   'lognormal', 'lognormal_clipped', 'lognormal_clipped_to_boundary',
                   'normal', 'normal_clipped', 'normal_clipped_to_boundary',
                   'poisson', 'poisson_clipped', 'poisson_clipped_to_boundary',
                   'uniform', 'uniform_int']


class NativeRNG(NativeRNG):
    """
    Signals that the random numbers will be drawn by NEST's own RNGs and 
    takes care of transforming pyNN parameters for the random distributions
    to NEST parameters.
    """
    translations = {
        'binomial':       {'n': 'n', 'p': 'p'},
        'gamma':          {'theta': 'scale', 'k': 'order'},
        'exponential':    {'beta': 'lambda'},
        'lognormal':      {'mu': 'mu', 'sigma': 'sigma'},
        'normal':         {'mu': 'mu', 'sigma': 'sigma'},
        'normal_clipped': {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'},
        'normal_clipped_to_boundary':
                          {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'},
        'poisson':        {'lambda_': 'lambda'},
        'uniform':        {'low': 'low', 'high': 'high'},
        'uniform_int':    {'low': 'low', 'high': 'high'},
        'vonmises':       {'mu': 'mu', 'kappa': 'kappa'},
    }

    def next(self, n=None, distribution=None, parameters=None, mask_local=None):
        # we ignore `n` and `mask_local`; they are needed for interface consistency
        parameter_map = self.translations[distribution]
        return NESTRandomDistribution(distribution,
                                      dict((parameter_map[k], v) for k, v in parameters.items()))


class NESTRandomDistribution(object):

    scale_parameters = {
        'gamma': ('scale',),
        'normal': ('mu', 'sigma'),
        'uniform': ('low', 'high'),
    }

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        self.shape = None  # pretend to lazyarray that I am an array

    def reshape(self, shape):
        # pretend to be an array
        return self

    def repr(self):
        D =  {'distribution': self.name}
        D.update(self.parameters)
        return D

    def __mul__(self, value):
        if not isinstance(value, Real):
            raise ValueError("Can only multiply by a number, not by a %s" % type(value))
        new_parameters = copy(self.parameters)
        if self.name in self.scale_parameters:
            for parameter_name in self.scale_parameters[self.name]:
                print("Multiplying parameter %s by %s" % (parameter_name, value))
                new_parameters[parameter_name] *=value
        else:
            raise NotImplementedError("Scaling not supported or not yet implemented for the %s distribution" % self.name)
        return NESTRandomDistribution(self.name, new_parameters)
