"""
Assorted utility classes and functions.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import os
import subprocess
import warnings
import numpy as np


def is_listlike(obj):
    """
    Check whether an object (a) can be converted into an array/list *and* has a
    length. This excludes iterators, for example.

    Maybe need to split into different functions, as don't always need length.
    """
    return (
        isinstance(obj, (list, tuple, set))
        or (isinstance(obj, np.ndarray) and obj.ndim > 0)
    )


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


def find(command):
    """Try to find an executable file."""
    path = os.environ.get("PATH", "").split(os.pathsep)
    cmd = ''
    for dir_name in path:
        abs_name = os.path.abspath(os.path.normpath(os.path.join(dir_name, command)))
        if os.path.isfile(abs_name):
            cmd = abs_name
            break
    return cmd


def run_command(path, working_directory):
    p = subprocess.Popen(path, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True,
                         cwd=working_directory)
    stdout, stderr = p.communicate()
    return p.returncode, stdout.split("\n")


class IndexBasedExpression(object):
    """
    Abstract base class for general expressions that use the cell indices and projection class to
    determine their value instead of just the the distance between the cells
    """

    @property
    def projection(self):
        try:
            return self._projection
        except AttributeError:
            return None

    @projection.setter
    def projection(self, projection):
        self._projection = projection

    def __call__(self, i, j):
        raise NotImplementedError
