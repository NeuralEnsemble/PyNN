"""
Conversion functions to nest-compatible data types.
"""

import numpy

from pyNN.parameters import Sequence

def make_sli_compatible_single(value):
    if isinstance(value, Sequence):
        return value.value
    else:
        return value


def make_sli_compatible(container):
    """
    Makes sure container only contains datatypes understood by the nest kernel.

    container can be scalar, a list or a dict.
    """

    compatible = None
    if isinstance(container, list):
        compatible = []
        for value in container:
            compatible.append(make_sli_compatible_single(value))

    elif isinstance(container, dict):
        compatible = {}

        for k,v in container.iteritems():
            compatible[k] = make_sli_compatible_single(v)

    else:
        compatible = make_sli_compatible_single(container)

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

