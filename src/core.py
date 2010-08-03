import numpy

def is_listlike(obj):
    return hasattr(obj, "__len__") and not isinstance(obj, basestring)

class LazyArray(object):
    """
    Optimises storage of arrays by storing only a single value if all the values
    in the array are the same.
    """
    
    def __init__(self, size, value):
        self.size = size
        self.value = value
        
    def __getitem__(self, i):
        if is_listlike(self.value):
            return self.value[i]
        else:
            return self.value
        
    def __setitem__(self, i, val):
        if val != self.value:
            self.value = self.value*numpy.ones((self.size,))
            self.value[i] = val
            