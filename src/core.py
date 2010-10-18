import numpy

def is_listlike(obj):
    """
    Check whether an object (a) can be converted into an array/list *and* has a
    length. This excludes iterators, for example.
    
    Maybe need to split into different functions, as don't always need length.
    """
    return hasattr(obj, "__len__") and not isinstance(obj, basestring)


class LazyArray(object):
    """
    Optimises storage of arrays by storing only a single value if all the values
    in the array are the same.
    """
    
    def __init__(self, size, value):
        assert isinstance(size, (int, long))
        #assert size > 1
        if is_listlike(value):
            assert numpy.isreal(value).all()
            assert size == len(value)
        else:
            assert numpy.isreal(value)
        self.size = size
        self.value = value
        
    def __getitem__(self, i):
        if not isinstance(i, (int, long)):
            raise TypeError("array indices must be integers, not %s" % type(i).__name__)
        if is_listlike(self.value):
            return self.value[i]
        else:
            if (i < -self.size) or (i >= self.size):
                raise IndexError("index out of bounds")
            return self.value
    
    def __setitem__(self, i, val):
        if is_listlike(self.value):
            self.value[i] = val
        elif val != self.value:
            if not is_listlike(self.value):
                self.value = self.value*numpy.ones((self.size,))
            self.value[i] = val
