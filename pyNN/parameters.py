"""
Parameter set handling

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections.abc import Sized
import numpy as np
from lazyarray import larray, partial_shape
from .core import is_listlike
from . import errors
from .random import RandomDistribution, NativeRNG


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
            may be an int, float, bool, NumPy array, iterator, generator
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
        if isinstance(value, str):
            err_msg = "Value should be a string expressing a function of d. "
            try:
                value = eval("lambda d: %s" % value)
            except SyntaxError:
                raise errors.InvalidParameterValueError(err_msg + "Incorrect syntax.")
            try:
                value(0.0)
            except NameError as err:
                raise errors.InvalidParameterValueError(err_msg + str(err))
        super(LazyArray, self).__init__(value, shape, dtype)

    def __setitem__(self, addr, new_value):
        self.check_bounds(addr)
        if (
            self.is_homogeneous
            and isinstance(new_value, (int, float, bool))
            and self.evaluate(simplify=True) == new_value
        ):
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
        column_indices = np.arange(self.ncols)
        if mask is not None:
            if not isinstance(mask, slice):
                assert len(mask) == self.ncols
            column_indices = column_indices[mask]
        if isinstance(self.base_value, RandomDistribution) and self.base_value.rng.parallel_safe:
            if mask is None:
                for j in column_indices:
                    yield self._partially_evaluate((slice(None), j), simplify=True)
            else:
                column_indices = np.arange(self.ncols)
                for j, local in zip(column_indices, mask):
                    col = self._partially_evaluate((slice(None), j), simplify=True)
                    if local:
                        yield col
        else:
            for j in column_indices:
                yield self._partially_evaluate((slice(None), j), simplify=True)

    def _apply_operations(self, x, addr=None, simplify=False):
        # todo: move this modified version back into lazyarray
        for f, arg in self.operations:
            if arg is None:
                x = f(x)
            elif isinstance(arg, larray):
                if addr is None:
                    x = f(x, arg.evaluate(simplify=simplify))
                else:
                    x = f(x, arg._partially_evaluate(addr, simplify=simplify))

            else:
                try:
                    x = f(x, arg)
                except TypeError:
                    assert isinstance(x, np.ndarray)
                    if x.dtype == np.dtype('O'):
                        x = np.array([f(xi, arg) for xi in x])
                    else:
                        raise
        return x


class ArrayParameter(object):
    """
    Represents a parameter whose value consists of multiple values, e.g. a tuple or array.

    The reason for defining this class rather than just using a NumPy array is
    to avoid the ambiguity of "is a given array a single parameter value (e.g.
    a spike train for one cell) or an array of parameter values (e.g. one number
    per cell)?".

    Arguments:
        `value`:
            anything which can be converted to a NumPy array, or another
            :class:`ArrayParameter` object.
    """

    def __init__(self, value):
        if isinstance(value, ArrayParameter):
            self.value = value.value
        elif isinstance(value, np.ndarray):
            # dont overwrite dtype of int arrays
            self.value = value
        else:
            self.value = np.array(value, float)

    # def __len__(self):
    #     This must not be defined, otherwise ArrayParameter is insufficiently different
    #     from NumPy array

    def max(self):
        """Return the maximum value."""
        return self.value.max()

    def __getitem__(self, item):
        return self.value[item]

    def __add__(self, val):
        """
        Return a new :class:`ArrayParameter` in which all values in the original
        :class:`ArrayParameter` have `val` added to them.

        If `val` is itself an array, return an array of :class:`ArrayParameter`
        objects, where ArrayParameter `i` is the original ArrayParameter added to
        element `i` of val.
        """
        if hasattr(val, '__len__'):
            # reshape if necessary?
            return np.array([self.__class__(self.value + x) for x in val], dtype=self.__class__)
        else:
            return self.__class__(self.value + val)

    def __sub__(self, val):
        """
        Return a new :class:`ArrayParameter` in which all values in the original
        :class:`ArrayParameter` have `val` subtracted from them.

        If `val` is itself an array, return an array of :class:`ArrayParameter`
        objects, where ArrayParameter `i` is the original ArrayParameter with
        element `i` of val subtracted from it.
        """
        if hasattr(val, '__len__'):
            # reshape if necessary?
            return np.array([self.__class__(self.value - x) for x in val], dtype=self.__class__)
        else:
            return self.__class__(self.value - val)

    def __mul__(self, val):
        """
        Return a new :class:`ArrayParameter` in which all values in the original
        :class:`ArrayParameter` have been multiplied by `val`.

        If `val` is itself an array, return an array of :class:`ArrayParameter`
        objects, where ArrayParameter `i` is the original ArrayParameter multiplied by
        element `i` of `val`.
        """
        if hasattr(val, '__len__') and not (hasattr(val, "shape") and len(val.shape) == 0):
            # the second condition weeds out 0-dimensional arrays, like Brian units.
            # reshape if necessary?
            return np.array([self.__class__(self.value * x) for x in val], dtype=self.__class__)
        else:
            return self.__class__(self.value * val)

    __rmul__ = __mul__

    def __div__(self, val):
        """
        Return a new :class:`ArrayParameter` in which all values in the original
        :class:`ArrayParameter` have been divided by `val`.

        If `val` is itself an array, return an array of :class:`ArrayParameter`
        objects, where ArrayParameter `i` is the original ArrayParameter divided by
        element `i` of `val`.
        """
        if hasattr(val, '__len__'):
            # reshape if necessary?
            return np.array([self.__class__(self.value / x) for x in val], dtype=self.__class__)
        else:
            return self.__class__(self.value / val)

    __truediv__ = __div__

    def __eq__(self, other):
        if isinstance(other, ArrayParameter):
            return self.value.size == other.value.size and (self.value == other.value).all()
        elif (
            isinstance(other, np.ndarray)
            and other.size > 0
            and isinstance(other[0], ArrayParameter)
        ):
            return np.array([(self == seq).all() for seq in other])
        else:
            return False

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.value)


class Sequence(ArrayParameter):
    """
        Represents a sequence of numerical values.

        Arguments:
            `value`:
                anything which can be converted to a NumPy array, or another
                :class:`Sequence` object.
    """
    # should perhaps use neo.SpikeTrain instead of this class, or at least allow a neo SpikeTrain
    pass


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
        self.children = {}
        self.schema = schema
        self._shape = shape
        self.component = component
        self.update(**parameters)
        self._evaluated = False

    def _set_shape(self, shape):
        for value in self._parameters.values():
            value.shape = shape
        for child in self.children.values():
            child.shape = shape
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
                    raise errors.NonExistentParameterError(
                            name,
                            model_name,
                            valid_parameter_names=self.schema.keys())
                if issubclass(expected_dtype, ArrayParameter) and isinstance(value, Sized):
                    if len(value) == 0:
                        value = ArrayParameter([])
                    elif not isinstance(value[0], ArrayParameter):
                        # may be a more generic way to do it, but for now this special-casing
                        # seems like the most robust approach
                        if isinstance(value[0], Sized):  # e.g. list of tuples
                            value = type(value)([ArrayParameter(x) for x in value])
                        else:
                            value = ArrayParameter(value)
                try:
                    self._parameters[name] = LazyArray(value, shape=self._shape,
                                                       dtype=expected_dtype)
                except (TypeError, errors.InvalidParameterValueError):
                    raise errors.InvalidParameterValueError(
                        f"For parameter {name} expected {expected_dtype}, got {type(value)}")
                except ValueError as err:
                    # maybe put the more specific error classes into lazyarray
                    raise errors.InvalidDimensionsError(err)
        else:
            for name, value in parameters.items():
                self._parameters[name] = LazyArray(value, shape=self._shape)

    def __getitem__(self, name):
        """x.__getitem__(y) <==> x[y]"""
        return self._parameters[name]

    def __setitem__(self, name, value):
        # need to add check against schema
        self._parameters[name] = value

    def pop(self, name, d=None):
        """
        Remove the given parameter from the parameter set and from its schema,
        and return its value.
        """
        value = self._parameters.pop(name, d)
        if self.schema:
            self.schema.pop(name, d)
        return value

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
                try:
                    if (
                        isinstance(value.base_value, RandomDistribution)
                        and value.base_value.rng.parallel_safe
                    ):
                        value = value.evaluate()  # can't partially evaluate if using parallel safe
                    self._parameters[name] = value[mask]
                except ValueError:
                    raise errors.InvalidParameterValueError(
                        f"{name} should not be of type {type(value)}")
            self._evaluated_shape = partial_shape(mask, self._shape)
        for child in self.children.values():
            child.evaluate(mask, simplify)
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
            assert not isinstance(D[name], LazyArray)  # should all have been evaluated by now
        for name, child in self.children.items():
            D[name] = child.as_dict()
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
                assert not isinstance(D[name], LazyArray)  # should all have been evaluated by now
            for name, child in self.children.items():
                D[name] = {}
                for cname, cvalue in child.items():
                    if is_listlike(cvalue):
                        D[name][cname] = cvalue[i]
                    else:
                        D[name][cname] = cvalue
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
                    # should all have been evaluated by now
                    assert not isinstance(D[name], LazyArray)
                yield D

    def __eq__(self, other):
        return (all(a == b for a, b in zip(self._parameters.items(), other._parameters.items()))
                and self.schema == other.schema
                and self._shape == other._shape)

    @property
    def parallel_safe(self):
        return any(
            isinstance(value.base_value, RandomDistribution) and value.base_value.rng.parallel_safe
            for value in self._parameters.values()
        )

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
        return (value for value in self._parameters.values()
                if isinstance(value.base_value, RandomDistribution))

    def expand(self, new_shape, mask):
        """
        Increase the size of the ParameterSpace.

        Existing array values are mapped to the indices given in mask.
        New array values are set to NaN.
        """
        for name, value in self._parameters.items():
            if isinstance(value.base_value, np.ndarray):
                new_base_value = np.ones(new_shape) * np.nan
                new_base_value[mask] = value.base_value
                self._parameters[name].base_value = new_base_value
        self.shape = new_shape

    def add_child(self, name, child_space):
        self.children[name] = child_space

    def flatten(self, with_prefix=True):
        for child_name, child in self.children.items():
            for name, value in child.items():
                if with_prefix:
                    self._parameters["{}.{}".format(child_name, name)] = value
                else:
                    self._parameters[name] = value
        self.children = {}


def simplify(value):
    """
    If `value` is a homogeneous array, return the single value that all elements
    share. Otherwise, pass the value through.
    """
    if isinstance(value, np.ndarray) and len(value.shape) > 0:
        #  latter condition is for Brian scalar quantities
        if (value == value[0]).all():
            return value[0]
        else:
            return value
    else:
        return value
    # alternative - need to benchmark
    # if np.any(arr != arr[0]):
    #    return arr
    # else:
    #    return arr[0]
