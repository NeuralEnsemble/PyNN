# -*- coding: utf-8 -*-
"""
Conversion functions to NEST-compatible data types.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy as np
from ..parameters import Sequence


def make_sli_compatible_single(value):
    if isinstance(value, Sequence):
        return_value = value.value
    elif isinstance(value, np.ndarray):
        if value.dtype == object and isinstance(value[0], Sequence):
            # check if the shape of the array is something other than (1,)
            # to my knowledge nest cannot handle that
            assert value.shape == (1,), "NEST expects 1 dimensional arrays"
            return_value = value[0].value
        elif value.shape == (1,):
            # for nest.SetDefaults, there is a difference between an (1,)-array
            # and a scalar value
            return_value = value[0]
        else:
            return_value = value
    else:
        return_value = value

    # nest does not understand numpy boolean values
    if isinstance(return_value, np.bool_):
        return_value = bool(return_value)

    return return_value


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

        for k, v in container.items():
            compatible[k] = make_sli_compatible_single(v)

    else:
        compatible = make_sli_compatible_single(container)

    return compatible


def make_pynn_compatible_single(value):
    # check if parameter is non-scalar
    if isinstance(value, np.ndarray):
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

        for k, v in container.items():
            compatible[k] = make_pynn_compatible_single(v)

    else:
        compatible = make_pynn_compatible_single(container)

    return compatible
