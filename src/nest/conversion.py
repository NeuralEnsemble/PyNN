"""
Conversion functions to nest-compatible data types.
"""

import numpy

from pyNN.parameters import Sequence

def make_sli_compatible_single(value, preserve_scalar_arrays=False):
    """
    preserve_scalar_arrays == True can be set to not convert shape (1,)-arrays
    to scalars.
    """
    if isinstance(value, Sequence):
        return_value = value.value
    elif isinstance(value, numpy.ndarray):
        if value.dtype == object and isinstance(value[0], Sequence):
            # check if the shape of the array is something other than (1,)
            # to my knowledge nest cannot handle that
            assert value.shape == (1,), "NEST expects 1 dimensional arrays"
            return_value = value[0].value
        elif value.shape == (1,) and not preserve_scalar_arrays:
            # for nest.SetDefaults, there is a difference between an (1,)-array
            # and a scalar value
            return_value = value[0]
        else:
            return_value = value
    else:
        return_value = value

    # nest does not understand numpy boolean values
    if isinstance(return_value, numpy.bool_):
        return_value = bool(return_value)

    return return_value


def make_sli_compatible(container, **kwargs):
    """
    Makes sure container only contains datatypes understood by the nest kernel.

    container can be scalar, a list or a dict.

    preserve_scalar_arrays == True can be set to not convert shape (1,)-arrays
    to scalars.
    """

    compatible = None
    if isinstance(container, list):
        compatible = []
        for value in container:
            compatible.append(make_sli_compatible_single(value, **kwargs))

    elif isinstance(container, dict):
        compatible = {}

        for k,v in container.iteritems():
            compatible[k] = make_sli_compatible_single(v, **kwargs)

    else:
        compatible = make_sli_compatible_single(container, **kwargs)

    return compatible


def make_pynn_compatible_single(value):
    # check if parameter is non-scalar
    if isinstance(value, numpy.ndarray):
        return Sequence(value)
    else:
        return value


def make_pynn_compatible(container):
    """
    Make sure that all entries in container do not confuse pyNN.

    container can be scalar, a list or a dict.
    """
    compatible = None
    if isinstance(container, list):
        compatible = []
        for value in container:
            compatible.append(make_pynn_compatible_single(value))

    elif isinstance(container, dict):
        compatible = {}

        for k,v in container.iteritems():
            compatible[k] = make_pynn_compatible_single(v)

    else:
        compatible = make_pynn_compatible_single(container)

    return compatible

