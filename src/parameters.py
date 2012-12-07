"""
Parameter set handling

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import collections
from pyNN.core import LazyArray, is_listlike
from pyNN import errors
from lazyarray import partial_shape


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
        """Return the maximum value from the sequence."""
        return self.value.max()

    def __mul__(self, val):
        """
        Return a new :class:`Sequence` in which all values in the original
        :class:`Sequence` have been multiplied by `val`.

        If `val` is itself an array, return an array of :class:`Sequence`
        objects, where sequence `i` is the original sequence multiplied by
        element `i` of `val`.
        """
        if hasattr(val, '__len__'):
            return numpy.array([Sequence(self.value * x) for x in val], dtype=Sequence) # reshape if necessary?
        else:
            return Sequence(self.value * val)

    __rmul__ = __mul__

    def __div__(self, val):
        """
        Return a new :class:`Sequence` in which all values in the original
        :class:`Sequence` have been divided by `val`.

        If `val` is itself an array, return an array of :class:`Sequence`
        objects, where sequence `i` is the original sequence divided by
        element `i` of `val`.
        """
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
        `shape`:
            the shape of the lazy arrays that will be constructed.

    .. _`lazy array`: http://readthedocs.org/docs/lazyarray/en/latest/index.html
    """

    def __init__(self, parameters, schema=None, shape=None, component=None):
        """

        """
        self._parameters = {}
        self.schema = schema
        self._shape = shape
        self.component = component
        self.update(**parameters)
        self._evaluated = False

    def _set_shape(self, shape):
        for value in self._parameters.itervalues():
            value.shape = shape
        self._shape = shape
    shape = property(fget=lambda self: self._shape, fset=_set_shape,
                     doc="Size of the lazy arrays contained within the parameter space")

    def keys(self):
        """
        PS.keys() -> list of PS's keys.
        """
        return self._parameters.keys()

    def items(self):
        """
        PS.items() ->  an iterator over the (key, value) items of PS.

        Note that the values will all be :class:`LazyArray` objects.
        """
        return self._parameters.iteritems()

    def __repr__(self):
        return "<ParameterSpace %s, shape=%s>" % (", ".join(self.keys()), self.shape)

    def update(self, **parameters):
        """
        Update the contents of the parameter space according to the
        `(key, value)` pairs in ``**parameters``. All values will be turned into
        lazy arrays.

        If the :class:`ParameterSpace` has a schema, the keys and the data types
        of the values will be checked against the schema.
        """
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
                    self._parameters[name] = LazyArray(value, shape=self._shape,
                                                       dtype=expected_dtype)
                except TypeError:
                    raise errors.InvalidParameterValueError("For parameter %s expected %s, got %s" % (name, type(value), expected_dtype))
                except ValueError as err:
                    raise errors.InvalidDimensionsError(err) # maybe put the more specific error classes into lazyarray
        else:
            for name, value in parameters.items():
                self._parameters[name] = LazyArray(value, shape=self._shape)

    def __getitem__(self, name):
        """x.__getitem__(y) <==> x[y]"""
        return self._parameters[name]

    @property
    def is_homogeneous(self):
        """
        True if all of the lazy arrays within are homogeneous.
        """
        return all(value.is_homogeneous for value in self._parameters.values())

    def evaluate(self, mask=None, simplify=False):
        """
        Evaluate all lazy arrays contained in the parameter space, using the
        given mask.
        """
        if self._shape is None:
            raise Exception("Must set shape of parameter space before evaluating")
        if mask is None:
            for name, value in self._parameters.items():
                self._parameters[name] = value.evaluate(simplify=simplify)
            self._evaluated_shape = self._shape
        else:
            for name, value in self._parameters.items():
                self._parameters[name] = value[mask]
            self._evaluated_shape = partial_shape(mask, self._shape)
        self._evaluated = True
        # should possibly update self.shape according to mask?

    def as_dict(self):
        """
        Return a plain dict containing the same keys and values as the
        parameter space. The values must first have been evaluated.
        """
        if not self._evaluated:
            raise Exception("Must call evaluate() method before calling ParameterSpace.as_dict()")
        D = {}
        for name, value in self._parameters.items():
            D[name] = value
            assert not isinstance(D[name], LazyArray) # should all have been evaluated by now
        return D

    def __iter__(self):
        r"""
        Return an array-element-wise iterator over the parameter space.

        Each item in the iterator is a dict, containing the same keys as the
        :class:`ParameterSpace`. For the `i`\th dict returned by the iterator,
        each value is the `i`\th element of the corresponding lazy array in the
        parameter space.

        Example:
        
        >>> ps = ParameterSpace({'a': [2, 3, 5, 8], 'b': 7, 'c': lambda i: 3*i+2}, shape=(4,))
        >>> ps.evaluate()
        >>> for D in ps:
        ...     print D
        ...
        {'a': 2, 'c': 2, 'b': 7}
        {'a': 3, 'c': 5, 'b': 7}
        {'a': 5, 'c': 8, 'b': 7}
        {'a': 8, 'c': 11, 'b': 7}
        """
        if not self._evaluated:
            raise Exception("Must call evaluate() method before iterating over a ParameterSpace")
        for i in range(self._evaluated_shape[0]):
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
                and self._shape == other._shape)


def simplify(value):
    """
    If `value` is a homogeneous array, return the single value that all elements
    share. Otherwise, pass the value through.
    """
    if isinstance(value, numpy.ndarray) and (value==value[0]).all():
        return value[0]
    else:
        return value
    # alternative - need to benchmark
    #if numpy.any(arr != arr[0]):
    #    return arr
    #else:
    #    return arr[0]
