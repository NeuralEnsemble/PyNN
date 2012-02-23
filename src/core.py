"""
Assorted utility classes and functions.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from pyNN import random
import functools
import warnings
from lazyarray import larray

def is_listlike(obj):
    """
    Check whether an object (a) can be converted into an array/list *and* has a
    length. This excludes iterators, for example.
    
    Maybe need to split into different functions, as don't always need length.
    """
    return isinstance(obj, (list, numpy.ndarray, tuple, set))


class LazyArray(larray):
    """
    Optimises storage of arrays in various ways:
      - stores only a single value if all the values in the array are the same
      - if the array is created from a RandomDistribution or a function f(i,j),
        then elements are only evaluated when they are accessed. Any operations
        performed on the array are also queued up to be executed on access.
        
    The main intention of the latter is to save memory for very large arrays by
    accessing them one row or column at a time: the entire array need never be
    in memory.
    """
    # most of the implementation moved to external lazyarray package
    # the plan is ultimately to move everything to lazyarray

    def __setitem__(self, addr, new_value):
        self.check_bounds(addr)
        if self.is_homogeneous and self.evaluate(simplify=True) == new_value:
            pass
        else:
            self.base_value = self.evaluate()
            self.base_value[addr] = new_value
            self.operations = []

    def by_column(self, mask=None):
        """
        Iterate over the columns of the array. Columns will be yielded either
        as a 1D array or as a single value (for a flat array).
        
        `mask`: either None or a boolean array indicating which columns should
                be included.
        """
        column_indices = numpy.arange(self.ncols)
        if mask is not None:
            assert len(mask) == self.ncols
            column_indices = column_indices[mask]
        if isinstance(self.base_value, random.RandomDistribution):
            if mask is None:
                for j in column_indices:
                    yield self._apply_operations(self.base_value.next(self.nrows, mask_local=False),
                                                 (slice(None), j))
            else:
                column_indices = numpy.arange(self.ncols)
                for j,local in zip(column_indices, mask):
                    col = self.base_value.next(self.nrows, mask_local=False)
                    if local:
                        yield self._apply_operations(col, (slice(None), j))
        else:
            for j in column_indices:
                yield self._partially_evaluate((slice(None), j), simplify=True)


# based on http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
class forgetful_memoize(object):
    """
    Decorator that caches the result from the last time a function was called.
    If the next call uses the same arguments, the cached value is returned, and
    not re-evaluated. If the next call uses different arguments, the cached
    value is overwritten.
    
    The use case is when the same, heavy-weight function is called repeatedly
    with the same arguments in different places.
    """
    
    def __init__(self, func):
        self.func = func
        self.cached_args = None
        self.cached_value = None
          
    def __call__(self, *args):
        import pdb; pdb.set_trace()
        if args == self.cached_args:
            print "using cached value"
            return self.cached_value
        else:
            #print "calculating value"
            value = self.func(*args)
            self.cached_args = args
            self.cached_value = value
            return value
    
    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


class deprecated(object):
    """
    Decorator to mark functions/methods as deprecated. Emits a warning when
    function is called and suggests a replacement.
    """
    
    def __init__(self, replacement=''):
        self.replacement = replacement
        
    def __call__(self, func):
        def new_func(*args, **kwargs):
            msg = "%s() is deprecated, and will be removed in a future release." % func.__name__
            if self.replacement:
                msg += " Use %s instead." % self.replacement
            warnings.warn(msg, category=DeprecationWarning)
            return func(*args, **kwargs)
        new_func.__name__ = func.__name__
        new_func.__doc__ = func.__doc__
        new_func.__dict__.update(func.__dict__)
        return new_func

        