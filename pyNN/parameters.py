"""
Parameter set handling

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:  # Python 2
    basestring
    long
except NameError:  # Python 3
    basestring = str
    long = int
import numpy
import collections
from pyNN.core import is_listlike
from pyNN import errors
from pyNN.random import RandomDistribution, NativeRNG
from lazyarray import larray, partial_shape


class LazyArray(larray):
    """
    Optimises storage of arrays in various ways:
      - stores only a single value if all the values in the array are the same
      - if the array is created from a :class:`~pyNN.random.RandomDistribution`
        or a function `f(i,j)`, then elements are only evaluated when they are
        accessed. Any operations performed on the array are also queued up to
        be executed on access.

    The main intention of the latter is to save memory for very large arrays by
    accessing them one row or column at a time: the entire array need never be
    in memory.

    Arguments:
        `value`:
            may be an int, long, float, bool, NumPy array, iterator, generator
            or a function, `f(i)` or `f(i,j)`, depending on the dimensions of
            the array. `f(i,j)` should return a single number when `i` and `j`
            are integers, and a 1D array when either `i` or `j` or both is a
            NumPy array (in the latter case the two arrays must have equal
            lengths).
        `shape`:
            a tuple giving the shape of the array, or `None`
        `dtype`:
            the NumPy `dtype`.
    """
    # most of the implementation moved to external lazyarray package
    # the plan is ultimately to move everything to lazyarray

    def __init__(self, value, shape=None, dtype=None):
        if isinstance(value, basestring):
            errmsg = "Value should be a string expressing a function of d. "
            try:
                value = eval("lambda d: %s" % value)
            except SyntaxError:
                raise errors.InvalidParameterValueError(errmsg + "Incorrect syntax.")
            try:
                value(0.0)
            except NameError as err:
                raise errors.InvalidParameterValueError(errmsg + str(err))
        super(LazyArray, self).__init__(value, shape, dtype)

    def __setitem__(self, addr, new_value):
        self.check_bounds(addr)
        if (self.is_homogeneous
            and isinstance(new_value, (int, long, float, bool))
            and self.evaluate(simplify=True) == new_value):
            pass
        else:
            self.base_value = self.evaluate()
            self.base_value[addr] = new_value
            self.operations = []

    def by_column(self, mask=None):
        """
        Iterate over the columns of the array. Columns will be yielded either
        as a 1D array or as a single value (for a flat array).

        `mask`: either `None` or a boolean array indicating which columns should be included.
        """
        column_indices = numpy.arange(self.ncols)
        if mask is not None:
            assert len(mask) == self.ncols
            column_indices = column_indices[mask]
        if isinstance(self.base_value, RandomDistribution) and self.base_value.rng.parallel_safe:
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
        elif isinstance(value, numpy.ndarray):
            # dont overwrite dtype of int arrays
            self.value = value
        else:
            self.value = numpy.array(value, float)

    #def __len__(self):
    #    This must not be defined, otherwise Sequence is insufficiently different from NumPy array

    def max(self):
        """Return the maximum value from the sequence."""
        return self.value.max()

    def __add__(self, val):
        """
        Return a new :class:`Sequence` in which all values in the original
        :class:`Sequence` have `val` added to them.

        If `val` is itself an array, return an array of :class:`Sequence`
        objects, where sequence `i` is the original sequence added to
        element `i` of val.
        """
        if hasattr(val, '__len__'):
            return numpy.array([Sequence(self.value + x) for x in val], dtype=Sequence) # reshape if necessary?
        else:
            return Sequence(self.value + val)

    def __sub__(self, val):
        """
        Return a new :class:`Sequence` in which all values in the original
        :class:`Sequence` have `val` subtracted from them.

        If `val` is itself an array, return an array of :class:`Sequence`
        objects, where sequence `i` is the original sequence with
        element `i` of val subtracted from it.
        """
        if hasattr(val, '__len__'):
            return numpy.array([Sequence(self.value - x) for x in val], dtype=Sequence) # reshape if necessary?
        else:
            return Sequence(self.value - val)

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

    .. _`lazy array`: https://lazyarray.readthedocs.org/
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
        for value in self._parameters.values():
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
        if hasattr(self._parameters, "iteritems"):
            return self._parameters.iteritems()
        else:
            return self._parameters.items()

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
                    if self.component:
                        model_name = self.component.__name__
                    else:
                        model_name = 'unknown'
                    raise errors.NonExistentParameterError(name,
                                                           model_name,
                                                           valid_parameter_names=self.schema.keys())
                if (expected_dtype == Sequence
                    and isinstance(value, collections.Sized)
                    and not isinstance(value[0], Sequence)): # may be a more generic way to do it, but for now this special-casing seems like the most robust approach
                    if isinstance(value[0], collections.Sized):  # e.g. list of tuples
                        value = type(value)([Sequence(x) for x in value])
                    else:
                        value = Sequence(value)
                try:
                    self._parameters[name] = LazyArray(value, shape=self._shape,
                                                       dtype=expected_dtype)
                except (TypeError, errors.InvalidParameterValueError):
                    raise errors.InvalidParameterValueError("For parameter %s expected %s, got %s" % (name, expected_dtype, type(value)))
                except ValueError as err:
                    raise errors.InvalidDimensionsError(err) # maybe put the more specific error classes into lazyarray
        else:
            for name, value in parameters.items():
                self._parameters[name] = LazyArray(value, shape=self._shape)

    def __getitem__(self, name):
        """x.__getitem__(y) <==> x[y]"""
        return self._parameters[name]

    def __setitem__(self, name, value):
        # need to add check against schema
        self._parameters[name] = value

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
                if isinstance(value.base_value, RandomDistribution) and value.base_value.rng.parallel_safe:
                    value = value.evaluate()  # can't partially evaluate if using parallel safe
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
        ...     print(D)
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

    def columns(self):
        """
        For a 2D space, return a column-wise iterator over the parameter space.
        """
        if not self._evaluated:
            raise Exception("Must call evaluate() method before iterating over a ParameterSpace")
        assert len(self.shape) == 2
        if len(self._evaluated_shape) == 1:  # values will be one-dimensional
            yield self._parameters
        else:
            for j in range(self._evaluated_shape[1]):
                D = {}
                for name, value in self._parameters.items():
                    if is_listlike(value):
                        D[name] = value[:, j]
                    else:
                        D[name] = value
                    assert not isinstance(D[name], LazyArray) # should all have been evaluated by now
                yield D

    def __eq__(self, other):
        return (all(a==b for a,b in zip(self._parameters.items(), other._parameters.items()))
                and self.schema == other.schema
                and self._shape == other._shape)

    @property
    def parallel_safe(self):
        return any(isinstance(value.base_value, RandomDistribution) and value.base_value.rng.parallel_safe
                   for value in self._parameters.values())

    @property
    def has_native_rngs(self):
        """
        Return True if the parameter set contains any NativeRNGs
        """
        return any(isinstance(rd.base_value.rng, NativeRNG)
                   for rd in self._random_distributions())

    def _random_distributions(self):
        """
        An iterator over those values contained in the PS that are
        derived from random distributions.
        """
        return (value for value in self._parameters.values() if isinstance(value.base_value, RandomDistribution))


def simplify(value):
    """
    If `value` is a homogeneous array, return the single value that all elements
    share. Otherwise, pass the value through.
    """
    if isinstance(value, numpy.ndarray):
        if (value==value[0]).all():
            return value[0]
        else:
            return value
    else:
        return value
    # alternative - need to benchmark
    #if numpy.any(arr != arr[0]):
    #    return arr
    #else:
    #    return arr[0]
