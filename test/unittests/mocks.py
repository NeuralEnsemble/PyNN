"""
Mock classes for unit tests

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
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
        s = self.start
        self.start += n*self.delta
        return s + self.delta*numpy.arange(n)
        
    def permutation(self, arr):
        return arr[::-1]