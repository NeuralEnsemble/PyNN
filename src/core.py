"""
Assorted utility classes and functions.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from pyNN import random
import operator
from copy import copy, deepcopy
import functools

def is_listlike(obj):
    """
    Check whether an object (a) can be converted into an array/list *and* has a
    length. This excludes iterators, for example.
    
    Maybe need to split into different functions, as don't always need length.
    """
    return type(obj) in [list, numpy.ndarray, tuple, set]


def check_shape(meth):
    """
    Decorator for LazyArray magic methods, to ensure that the operand has
    the same shape as the array.
    """
    def wrapped_meth(self, val):
        if isinstance(val, (LazyArray, numpy.ndarray)):
            if val.shape != self.shape:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        return meth(self, val)
    return wrapped_meth

def lazy_operation(name):
    def op(self, val):
        new_map = deepcopy(self)
        new_map.operations.append((getattr(operator, name), val))
        return new_map
    return check_shape(op)

def lazy_inplace_operation(name):
    def op(self, val):
        self.operations.append((getattr(operator, name), val))
        return self
    return check_shape(op)

def lazy_unary_operation(name):
    def op(self):
        new_map = deepcopy(self)
        new_map.operations.append((getattr(operator, name), None))
        return new_map
    return op


class LazyArray(object):
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

    def __init__(self, value, shape):
        """
        Create a new LazyArray.
        
        `value` : may be an int, long, float, bool, numpy array,
                  RandomDistribution or a function f(i,j).
                  
        f(i,j) should return a single number when i and j are integers, and a 1D
        array when either i or j is a numpy array. The case where both i and j
        are arrays need not be supported.
        """
        if is_listlike(value):
            assert numpy.isreal(value).all()
            if not isinstance(value, numpy.ndarray):
                value = numpy.array(value)
            assert value.shape == shape, "Array has shape %s, value has shape %s" % (shape, value.shape) # this should be true even when using MPI
        else:
            assert numpy.isreal(value)
        self.base_value = value
        self.shape = shape
        self.operations = []
        
    @property
    def nrows(self):
        return self.shape[0]
    
    @property
    def ncols(self):
        return self.shape[1]
    
    def __getitem__(self, addr):
        if isinstance(addr, (int, long, float)):
            addr = (addr,)
        if len(addr) != len(self.shape):
            raise IndexError("invalid index")
        if not isinstance(addr, (int, long, tuple)):
            raise TypeError("array indices must be integers, not %s" % type(addr).__name__)
        val = self.value
        if isinstance(val, (int, long, float)):
            self.check_bounds(addr)
            return val
        else:
            return val[addr]
    
    def __setitem__(self, addr, new_value):
        self.check_bounds(addr)
        val = self.value
        if isinstance(val, (int, long, float)) and val == new_value:
            pass
        else:
            self.base_value = self.as_array()
            self.base_value[addr] = new_value
    
    def check_bounds(self, addr):
        if isinstance(addr, (int, long, float)):
            addr = (addr,)
        for i, size in zip(addr, self.shape):
            if (i < -size) or (i >= size):
                raise IndexError("index out of bounds")
    
    def apply(self, f):
        """
        Add the function f(x) to the list of the operations to be performed,
        where x will be a scalar or a numpy array.
        
        >>> m = LazyArray(4, shape=(2,2))
        >>> m.apply(numpy.sqrt)
        >>> m.value
        2.0
        >>> m.as_array()
        array([[ 2.,  2.],
               [ 2.,  2.]])
        """
        self.operations.append((f, None))
    
    def _apply_operations(self, x):
        for f, arg in self.operations:
            if arg is None:
                x = f(x)
            else:
                x = f(x, arg)
        return x
    
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
        if isinstance(self.base_value, (int, long, float, bool)):
            for j in column_indices:
                yield self._apply_operations(self.base_value)
        elif isinstance(self.base_value, numpy.ndarray):
            for j in column_indices:
                yield self._apply_operations(self.base_value[:, j])
        elif isinstance(self.base_value, random.RandomDistribution):
            if mask is None:
                for j in column_indices:
                    yield self._apply_operations(self.base_value.next(self.nrows, mask_local=False))
            else:
                column_indices = numpy.arange(self.ncols)
                for j,local in zip(column_indices, mask):
                    col = self.base_value.next(self.nrows, mask_local=False)
                    if local:
                        yield self._apply_operations(col)
        #elif isinstance(self.base_value, LazyArray):
        #    for column in self.base_value.by_column(mask=mask):
        #        yield self._apply_operations(column)
        elif callable(self.base_value): # a function of (i,j)
            row_indices = numpy.arange(self.nrows, dtype=int)
            for j in column_indices:
                yield self._apply_operations(self.base_value(row_indices, j))
        else:
            raise Exception("invalid mapping")

    @property
    def value(self):
        """
        Returns the base value with all operations applied to it. Works only
        when the base value is a scalar or a real numpy array, not when the
        base value is a RandomDistribution or mapping function.
        """
        val = self._apply_operations(self.base_value)
        if isinstance(val, LazyArray):
            val = val.value
        return val

    def as_array(self):
        """
        Return the LazyArray as a real numpy array.
        """
        if isinstance(self.base_value, (int, long, float, bool)):
            x = self.base_value*numpy.ones(self.shape)
        elif isinstance(self.base_value, numpy.ndarray):
            x = self.base_value
        elif isinstance(self.base_value, random.RandomDistribution):
            n = self.nrows*self.ncols
            x = self.base_value.next(n).reshape(self.shape)
        elif callable(self.base_value):
            row_indices = numpy.arange(self.nrows, dtype=int)
            x = numpy.array([self.base_value(row_indices, j) for j in range(self.ncols)]).T
        else:
            raise Exception("invalid mapping")
        return self._apply_operations(x)

    __iadd__ = lazy_inplace_operation('add')
    __isub__ = lazy_inplace_operation('sub')
    __imul__ = lazy_inplace_operation('mul')
    __idiv__ = lazy_inplace_operation('div')
    __ipow__  = lazy_inplace_operation('pow')
    
    __add__  = lazy_operation('add')
    __radd__ = __add__
    __sub__  = lazy_operation('sub')
    __mul__  = lazy_operation('mul')
    __rmul__ = __mul__
    __div__  = lazy_operation('div')
    __pow__  = lazy_operation('pow')
    
    __lt__   = lazy_operation('lt')
    __gt__   = lazy_operation('gt')
    __le__   = lazy_operation('le')
    __ge__   = lazy_operation('ge')
    
    __neg__  = lazy_unary_operation('neg')
    __pos__  = lazy_unary_operation('pos')
    __abs__  = lazy_unary_operation('abs')


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
