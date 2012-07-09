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

    Arguments:
        `value`:
            anything which can be converted to a NumPy array, or another
            :class:`Sequence` object.
    """
    # should perhaps use neo.SpikeTrain and neo.EventArray instead of this class?

    def __init__(self, value):
        if isinstance(value, Sequence):
            self.value = value.value
        else:
            self.value = numpy.array(value, float)

    #def __len__(self):
    #    This must not be defined, otherwise Sequence is insufficiently different from NumPy array

    def max(self):
        """docstring goes here"""
        return self.value.max()

    def __mul__(self, val):
        """docstring goes here"""
        if hasattr(val, '__len__'):
            return numpy.array([Sequence(self.value * x) for x in val], dtype=Sequence) # reshape if necessary?
        else:
            return Sequence(self.value * val)

    __rmul__ = __mul__

    def __div__(self, val):
        """docstring goes here"""
        if hasattr(val, '__len__'):
            return numpy.array([Sequence(self.value/x) for x in val], dtype=Sequence) # reshape if necessary?
        else:
            return Sequence(self.value/val)

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return self.value.size == other.value.size and (self.value == other.value).all()
        elif isinstance(other, numpy.ndarray) and other.size > 0 and isinstance(other[0], Sequence):
            return numpy.array([(self == seq).all() for seq in other])
        else:
            return False

    def __repr__(self):
        return "Sequence(%s)" % self.value


class ParameterSpace(object):
    """
    Representation of one or more points in a parameter space.

    i.e. represents one or more parameter sets, where each parameter set has
    the same parameter names and types but the parameters may have different
    values.

    Arguments:
        `parameters`:
            a dict containing values of any type that may be used to construct a
            `lazy array`_, i.e. `int`, `float`, NumPy array,
            :class:`~pyNN.random.RandomDistribution`, function that accepts a
            single argument.
        `schema`:
            a dict whose keys are the expected parameter names and whose values
            are the expected parameter types
        `component`:
            optional - class for which the parameters are destined. Used in
            error messages.

    .. _`lazy array`: http://readthedocs.org/docs/lazyarray/en/latest/index.html
    """

    def __init__(self, parameters, schema=None, size=None, component=None):
        """

        """
        self._parameters = {}
        self.schema = schema
        self._size = size
        self.component = component
        self.update(**parameters)
        self._evaluated = False

    def _set_size(self, n):
        for value in self._parameters.itervalues():
            value.shape = (n,)
        self._size = n
    size = property(fget=lambda self: self._size, fset=_set_size)

    def keys(self):
        """
        docstring goes here
        """
        return self._parameters.keys()

    def items(self):
        """
        docstring goes here
        """
        return self._parameters.iteritems()

    def __repr__(self):
        return "<ParameterSpace %s, size=%s>" % (", ".join(self.keys()), self.size)

    def update(self, **parameters):
        """
        docstring goes here
        """
        if self._size is None:
            array_shape = None
        else:
            array_shape = (self._size,)
        if self.schema:
            for name, value in parameters.items():
                try:
                    expected_dtype = self.schema[name]
                except KeyError:
                    raise errors.NonExistentParameterError(name,
                                                           self.component.__name__,
                                                           valid_parameter_names=self.schema.keys())
                if (expected_dtype == Sequence
                    and isinstance(value, collections.Sized)
                    and not isinstance(value[0], Sequence)): # may be a more generic way to do it, but for now this special-casing seems like the most robust approach
                    value = Sequence(value)
                try:
                    self._parameters[name] = LazyArray(value, shape=array_shape,
                                                       dtype=expected_dtype)
                except TypeError:
                    raise errors.InvalidParameterValueError("For parameter %s expected %s, got %s" % (name, type(value), expected_dtype))
                except ValueError as err:
                    raise errors.InvalidDimensionsError(err) # maybe put the more specific error classes into lazyarray
        else:
            for name, value in parameters.items():
                self._parameters[name] = LazyArray(value, shape=array_shape)

    def __getitem__(self, name):
        return self._parameters[name]

    @property
    def is_homogeneous(self):
        """
        docstring goes here
        """
        return all(value.is_homogeneous for value in self._parameters.values())

    def evaluate(self, mask=None, simplify=False):
        """
        Evaluate all lazy arrays contained in the parameter space, using the
        given mask.
        """
        if self._size is None:
            raise Exception("Must set size of parameter space before evaluating")
        if mask is None:
            for name, value in self._parameters.items():
                self._parameters[name] = value.evaluate(simplify=simplify)
            self._evaluated_size = self._size
        else:
            if len(mask) > 0:
                for name, value in self._parameters.items():
                    self._parameters[name] = value[mask]
            self._evaluated_size = len(mask)
        self._evaluated = True
        # should possibly update self.size according to mask?

    def as_dict(self):
        """
        docstring goes here
        """
        if not self._evaluated:
            raise Exception("Must call evaluate() method before calling ParameterSpace.as_dict()")
        D = {}
        for name, value in self._parameters.items():
            D[name] = value
            assert not isinstance(D[name], LazyArray) # should all have been evaluated by now
        return D

    def __iter__(self):
        """
        docstring goes here
        """
        if not self._evaluated:
            raise Exception("Must call evaluate() method before iterating over a ParameterSpace")
        for i in range(self._evaluated_size):
            D = {}
            for name, value in self._parameters.items():
                if is_listlike(value):
                    D[name] = value[i]
                else:
                    D[name] = value
                assert not isinstance(D[name], LazyArray) # should all have been evaluated by now
            yield D

    def __eq__(self, other):
        return (all(a==b for a,b in zip(self._parameters.items(), other._parameters.items()))
                and self.schema == other.schema
                and self._size == other._size)

def simplify(value):
    """
    If `value` is a homogeneous array, return the single value that all elements
    share. Otherwise, pass the value through.
    """
    if isinstance(value, numpy.ndarray) and (value==value[0]).all():
        return value[0]
    else:
        return value
