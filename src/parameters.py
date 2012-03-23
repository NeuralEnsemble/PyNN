"""
Parameter set handling

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import collections
from pyNN.core import LazyArray, is_listlike
from pyNN import errors


class Sequence(object):
    """
    Represents a sequence of numerical values.
    
    The reason for defining this class rather than just using a NumPy array is
    to avoid the ambiguity of "is a given array a single parameter value (e.g.
    a spike train for one cell) or an array of parameter values (e.g. one number
    per cell)?".
    """
     
    def __init__(self, value):
        if isinstance(value, Sequence):
            self.value = value.value
        else:
            self.value = numpy.array(value, float)

    #def __len__(self):
    #    This must not be defined, otherwise Sequence is insufficiently different from NumPy array
        
    def max(self):
        return self.value.max()

    def __mul__(self, val):
        if hasattr(val, '__len__'):
            return numpy.array([Sequence(self.value * x) for x in val], dtype=Sequence) # reshape if necessary?
        else:
            return Sequence(self.value * val)


class ParameterSpace(object):
    """
    Representation of one or more points in a parameter space.
    
    i.e. represents one or more parameter sets, where each parameter set has
    the same parameter names and types but the parameters may have different
    values.
    """
    
    def __init__(self, parameters, schema, size):
        """
        `parameters` - a dict containing values of any type that may be used to
                       construct a lazy array, i.e. int, float, NumPy array,
                       RandomDistribution, function that accepts a single
                       argument.
        `schema` - a dict whose keys are the expected parameter names and whose
                   values are the expected parameter types
        """
        self._parameters = {}
        self.schema = schema
        self._size = size
        self.update(**parameters)
        self._evaluated = False

    def _set_size(self, n):
        for value in self._parameters.itervalues():
            value.shape = (n,)
        self._size = n
    size = property(fget=lambda self: self._size, fset=_set_size)

    def keys(self):
        return self._parameters.keys()

    def update(self, **parameters):
        if self.schema:
            for name, value in parameters.items():
                expected_dtype = self.schema[name]
                if expected_dtype == Sequence and isinstance(value, collections.Sized): # may be a more generic way to do it, but for now this special-casing seems like the most robust approach
                    value = Sequence(value)
                self._parameters[name] = LazyArray(value, shape=(self._size,),
                                                   dtype=expected_dtype)
        else:
            for name, value in parameters.items():
                self._parameters[name] = LazyArray(value, shape=(self._size,))
    
    def __getitem__(self, name):
        return self._parameters[name]
        
    @property
    def is_homogeneous(self):
        return all(value.is_homogeneous for value in self._parameters.values())
    
    def evaluate(self, mask=None, simplify=False):
        """
        Evaluate all lazy arrays contained in the parameter space, using the
        given mask.
        """
        if self._size is None:
            raise Exception("Must set size of parameter space before evaluating")
        if mask:
            for name, value in self._parameters:
                self._parameters[name] = value[mask]
        else:
            for name, value in self._parameters.items():
                self._parameters[name] = value.evaluate(simplify=simplify)
        self._evaluated = True
        # should possibly update self.size according to mask?

    def as_dict(self):
        if not self._evaluated:
            raise Exception("Must call evaluate() method before calling ParameterSpace.as_dict()")
        D = {}
        for name, value in self._parameters.items():
            assert not is_listlike(value)
            D[name] = value
            assert not isinstance(D[name], LazyArray) # should all have been evaluated by no
        return D

    def __iter__(self):
        if not self._evaluated:
            raise Exception("Must call evaluate() method before iterating over a ParameterSpace")
        for i in range(self._size):
            D = {}
            for name, value in self._parameters.items():
                if is_listlike(value):
                    D[name] = value[i]
                else:
                    D[name] = value
                assert not isinstance(D[name], LazyArray) # should all have been evaluated by now
            yield D
