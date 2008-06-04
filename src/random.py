"""
Provides wrappers for several random number generators, giving them all a
common interface so that they can be used interchangeably in PyNN.

Note however that we have so far made no effort to implement parameter
translation, and parameter names/order may be different for the different RNGs.

$Id:random.py 188 2008-01-29 10:03:59Z apdavison $
"""

import sys
import numpy.random
try:
    import pygsl.rng
except ImportError:
    print "Warning: GSL random number generators not available"
import time
import logging

# The following two functions taken from
# http://www.nedbatchelder.com/text/pythonic-interfaces.html
def _function_id(obj, n_frames_up):
    """ Create a string naming the function n frames up on the stack. """
    frame = sys._getframe(n_frames_up+1)
    code = frame.f_code
    return "%s.%s" % (obj.__class__, code.co_name)
 
def abstractMethod(obj=None):
    """ Use this instead of 'pass' for the body of abstract methods. """
    raise Exception("Unimplemented abstract method: %s" % _function_id(obj, 1))
 
 
class AbstractRNG:
    """Abstract class for wrapping random number generators. The idea is to be
    able to use either simulator-native rngs, which may be more efficient, or a
    standard python rng, e.g. a numpy.random.RandomState object, which would
    allow the same random numbers to be used across different simulators, or
    simply to read externally-generated numbers from files."""
    
    def __init__(self, seed=None):
        if seed:
            assert isinstance(seed, int), "`seed` must be an int (< %d), not a %s" % (sys.maxint, type(seed).__name__)
        self.seed = seed
        # define some aliases
        self.random = self.next
        self.sample = self.next
    
    def next(self, n=1, distribution='uniform', parameters=[]):
        """Return n random numbers from the distribution.
        
        If n is 1, return a float, if n > 1, return a numpy array,
        if n <= 0, raise an Exception."""
        abstractMethod(self)

    
class NumpyRNG(AbstractRNG):
    """Wrapper for the numpy.random.RandomState class (Mersenne Twister PRNG)."""
    
    def __init__(self, seed=None, rank=0, num_processes=1, parallel_safe=False):
        AbstractRNG.__init__(self, seed)
        self.rng = numpy.random.RandomState()
        if self.seed:
            if not parallel_safe:
                self.seed += rank # ensure different nodes get different sequences
                if rank != 0:
                    logging.warning("Changing the seed to %s on node %d" % (self.seed, rank))
            self.rng.seed(self.seed)
        else:
            self.rng.seed()
        self.rank = rank # MPI rank
        self.num_processes = num_processes # total number of MPI processes
        self.parallel_safe = parallel_safe
            
    def __getattr__(self, name):
        """This is to give NumpyRNG the same methods as numpy.random.RandomState."""
        return getattr(self.rng, name)
    
    def next(self, n=1, distribution='uniform', parameters=[]):
        """Return n random numbers from the distribution.
        
        If n >= 0, return a numpy array,
        if n < 0, raise an Exception."""      
        if n == 0:
            rarr = numpy.random.rand(0) # We return an empty array
        elif n > 0:
            if self.num_processes > 1 and not self.parallel_safe:
                # n is the number for the whole model, so if we do not care about
                # having exactly the same random numbers independent of the
                # number of processors (m), we only need generate n/m+1 per node
                # (assuming round-robin distribution of cells between processors)
                n = n/self.num_processes + 1 
            rarr = getattr(self.rng, distribution)(size=n, *parameters)
        else:
            raise ValueError, "The sample number must be positive"
        if self.parallel_safe and self.num_processes > 1:
            # strip out the random numbers that should be used on other processors.
            # This assumes that the first neuron in a population is always created on
            # the node with rank 0, and that neurons are distributed in a round-robin
            # This assumption is not true for NEST
            rarr = rarr[numpy.arange(self.rank, len(rarr), self.num_processes)]
        return rarr

class GSLRNG(AbstractRNG):
    """Wrapper for the GSL random number generators."""
       
    def __init__(self, seed=None, type='mt19937'):
        AbstractRNG.__init__(self, seed)
        self.rng = getattr(pygsl.rng, type)()
        if self.seed  :
            self.rng.set(self.seed)
        else:
            self.seed = int(time.time())
            self.rng.set(self.seed)
    
    def __getattr__(self, name):
        """This is to give GSLRNG the same methods as the GSL RNGs."""
        return getattr(self.rng, name)
    
    def next(self, n=1, distribution='uniform', parameters=[]):
        """Return n random numbers from the distribution.
        
        If n is 1, return a float, if n > 1, return a numpy array,
        if n < 0, raise an Exception."""
        p = parameters + [n]
        if n == 0:
            return numpy.random.rand(0) # We return an empty array
        if n > 0:
            return getattr(self.rng, distribution)(*p)
        else:
            raise ValueError, "The sample number must be positive"


    
class NativeRNG(AbstractRNG):
    """Signals that the simulator's own native RNG should be used.
    Each simulator module should implement a class of the same name which
    inherits from this and which sets the seed appropriately."""
    pass


class RandomDistribution:
    """Class which defines a next(n) method which returns an array of n random
       numbers from a given distribution."""
       
    def __init__(self, distribution='uniform', parameters=[], rng=None):
        """
        If present, rng should be a NumpyRNG or GSLRNG object.
        distribution should be the name of a method supported by the underlying
            random number generator object.
        parameters should be a list or tuple containing the arguments expected
            by the underlying method in the correct order. named arguments are
            not yet supported.
        Note that NumpyRNG and GSLRNG distributions may not have the same names,
            e.g., 'normal' for NumpyRNG and 'gaussian' for GSLRNG, and the
            arguments may also differ.
        """ 
        self.name = distribution
        assert isinstance(parameters, (list, tuple, dict)), "The parameters argument must be a list or tuple or dict"
        self.parameters = parameters
        if rng:
            assert isinstance(rng, AbstractRNG), "rng must be a pyNN.random RNG object"
            self.rng = rng
        else: # use numpy.random.RandomState() by default
            self.rng = NumpyRNG()
        
    def next(self, n=1):
        """Return n random numbers from the distribution."""
        return self.rng.next(n=n,
                             distribution=self.name,
                             parameters=self.parameters)
        
