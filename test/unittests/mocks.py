"""
Mock classes for unit tests

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import random
import numpy


class MockRNG(random.WrappedRNG):
    rng = None

    def __init__(self, start=0.0, delta=1, parallel_safe=True):
        random.WrappedRNG.__init__(self, parallel_safe=parallel_safe)
        self.start = start
        self.delta = delta

    def _next(self, distribution, n, parameters):
        if distribution == "uniform_int":
            return self._next_int(n, parameters)
        elif distribution == "binomial":
            return self._next_binomial(n, parameters)
        s = self.start
        self.start += n*self.delta
        return s + self.delta*numpy.arange(n)

    def _next_int(self, n, parameters):
        low, high = parameters["low"], parameters["high"]
        s = int(self.start)
        self.start += n*self.delta
        x = s + self.delta*numpy.arange(n)
        return x % (high - low) + low

    def _next_binomial(self, n, parameters):
        return self._next_int(n, {"low": 0, "high": parameters["n"]})

    def permutation(self, arr):
        return arr[::-1]


class MockRNG2(random.WrappedRNG):
    rng = None

    def __init__(self, numbers, parallel_safe=True):
        random.WrappedRNG.__init__(self, parallel_safe=parallel_safe)
        self.numbers = numbers
        self.i = 0

    def _next(self, distribution, n, parameters):
        x = self.numbers[self.i:self.i+n]
        self.i += n
        return x

    def permutation(self, arr):
        return arr[::-1]
    
class MockRNG3(random.WrappedRNG):
    """
    returns [1, 0, 0, 0,..]
    """
    rng = None

    def __init__(self, parallel_safe=True):
        random.WrappedRNG.__init__(self, parallel_safe=parallel_safe)

    def _next(self, distribution, n, parameters):
        x = numpy.zeros(n)
        x.dtype=int
        x[0]=1
        return x
    
    def permutation(self, arr):
        return arr[::-1]
