import numpy
from pyNN import random
import operator
from copy import deepcopy

def is_listlike(obj):
    """
    Check whether an object (a) can be converted into an array/list *and* has a
    length. This excludes iterators, for example.
    
    Maybe need to split into different functions, as don't always need length.
    """
    return hasattr(obj, "__len__") and not isinstance(obj, basestring)


def check_shape(meth):
    def wrapped_meth(self, val):
        if isinstance(val, (LazyArray, numpy.ndarray)):
            if val.shape != self.shape:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        return meth(self, val)
    return wrapped_meth

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
        if is_listlike(value):
            assert numpy.isreal(value).all()
            if not isinstance(value, numpy.ndarray):
                value = numpy.array(value)
            assert value.shape == shape
        else:
            assert numpy.isreal(value)
        self.initial_value = value
        self.shape = shape
        self.operations = []
        
    @property
    def nrows(self):
        return self.shape[0]
    
    @property
    def ncols(self):
        return self.shape[1]
    
    def __getitem__(self, addr):
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
            self.initial_value = self.as_array()
            self.initial_value[addr] = new_value
    
    def check_bounds(self, addr):
        if isinstance(addr, (int, long, float)):
            addr = (addr,)
        for i, size in zip(addr, self.shape):
            if (i < -size) or (i >= size):
                raise IndexError("index out of bounds")
    
    def _apply_operations(self, x):
        for f, arg in self.operations:
            x = f(x, arg)
        return x
    
    def by_column(self, mask=None):
        column_indices = numpy.arange(self.ncols)
        if mask is not None:
            assert len(mask) == self.ncols
            column_indices = column_indices[mask]
        if isinstance(self.initial_value, (int, long, float, bool)):
            for j in column_indices:
                yield self._apply_operations(self.initial_value)
        elif isinstance(self.initial_value, numpy.ndarray):
            for j in column_indices:
                yield self._apply_operations(self.initial_value[:, j])
        elif isinstance(self.initial_value, random.RandomDistribution):
            if mask is None:
                for j in column_indices:
                    yield self._apply_operations(self.initial_value.next(self.nrows, mask_local=False))
            else:
                column_indices = numpy.arange(self.ncols)
                for j,local in zip(column_indices, mask):
                    col = self.initial_value.next(self.nrows, mask_local=False)
                    if local:
                        yield self._apply_operations(col)
        elif callable(self.initial_value): # a function of (i,j)
            for j in column_indices:
                row_indices = numpy.arange(self.nrows, dtype=int)
                yield self._apply_operations(self.initial_value(row_indices, j))
        else:
            raise Exception("invalid mapping")

    @property
    def value(self):
        val = self._apply_operations(self.initial_value)
        if isinstance(val, LazyArray):
            val = val.value
        return val

    def as_array(self):
        if isinstance(self.initial_value, (int, long, float, bool)):
            x = self.initial_value*numpy.ones(self.shape)
        elif isinstance(self.initial_value, numpy.ndarray):
            x = self.initial_value
        elif isinstance(self.initial_value, random.RandomDistribution):
            n = self.nrows*self.ncols
            x = self.initial_value.next(n).reshape(self.shape)
        elif callable(self.initial_value):
            j, i = numpy.meshgrid(range(self.ncols), range(self.nrows))
            x = self.initial_value(i, j)
        else:
            raise Exception("invalid mapping")
        return self._apply_operations(x)

    @check_shape
    def __iadd__(self, val):
        self.operations.append((operator.add, val))
        return self

    @check_shape
    def __add__(self, val):
        new_map = deepcopy(self)
        new_map.operations.append((operator.add, val))
        return new_map
    __radd__ = __add__

    @check_shape
    def __mul__(self, val):
        new_map = deepcopy(self)
        new_map.operations.append((operator.mul, val))
        return new_map
    __rmul__ = __mul__

    @check_shape
    def __lt__(self, val):
        new_map = deepcopy(self)
        new_map.operations.append((operator.lt, val))
        return new_map
