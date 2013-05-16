"""
Assorted utility classes and functions.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import warnings
import numpy
from pyNN import errors


def is_listlike(obj):
    """
    Check whether an object (a) can be converted into an array/list *and* has a
    length. This excludes iterators, for example.

    Maybe need to split into different functions, as don't always need length.
    """
    return isinstance(obj, (list, numpy.ndarray, tuple, set))


class deprecated(object):
    """
    Decorator to mark functions/methods as deprecated. Emits a warning when
    function is called and suggests a replacement.
    """

    def __init__(self, replacement=''):
        self.replacement = replacement

    def __call__(self, func):
        def new_func(*args, **kwargs):
            msg = "%s() is deprecated, and will be removed in a future release." % func.__name__
            if self.replacement:
                msg += " Use %s instead." % self.replacement
            warnings.warn(msg, category=DeprecationWarning)
            return func(*args, **kwargs)
        new_func.__name__ = func.__name__
        new_func.__doc__ = "*Deprecated*. Use ``%s`` instead." % self.replacement
        new_func.__dict__.update(func.__dict__)
        return new_func

def reraise(exception, message):
    args = list(exception.args)
    args[0] += message
    exception.args = args
    raise

def ezip(*args):
    for items in zip(*args):
        yield items[0], items[1:]
