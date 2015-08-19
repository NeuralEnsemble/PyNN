# encoding: utf-8
"""
A collection of utility functions and classes.

Functions:
    notify()          - send an e-mail when a simulation has finished.
    get_script_args() - get the command line arguments to the script, however
                        it was run (python, nrniv, mpirun, etc.).
    init_logging()    - convenience function for setting up logging to file and
                        to the screen.

    Timer    - a convenience wrapper around the time.time() function from the
               standard library.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from __future__ import print_function
# If there is a settings.py file on the path, defaults will be
# taken from there.
try:
    from settings import SMTPHOST, EMAIL
except ImportError:
    SMTPHOST = None
    EMAIL = None
try:
    unicode
except NameError:
    unicode = str
import sys
import logging
import time
import os
from datetime import datetime
import functools
import numpy
try:
    from importlib import import_module
except ImportError:  # Python 2.6
    def import_module(name):
        return __import__(name)
    
from pyNN.core import deprecated


def notify(msg="Simulation finished.", subject="Simulation finished.",
           smtphost=SMTPHOST, address=EMAIL):
    """Send an e-mail stating that the simulation has finished."""
    if not (smtphost and address):
        print("SMTP host and/or e-mail address not specified.\nUnable to send notification message.")
    else:
        import smtplib, datetime
        msg = ("From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n") % (address,address,subject) + msg
        msg += "\nTimestamp: %s" % datetime.datetime.now().strftime("%H:%M:%S, %F")
        server = smtplib.SMTP(smtphost)
        server.sendmail(address, address, msg)
        server.quit()


def get_script_args(n_args, usage=''):
    """
    Get command line arguments.

    This works by finding the name of the main script and assuming any
    arguments after this in sys.argv are arguments to the script.
    It would be nicer to use optparse, but this doesn't seem to work too well
    with nrniv or mpirun.
    """
    calling_frame = sys._getframe(1)
    if '__file__' in calling_frame.f_locals:
        script = calling_frame.f_locals['__file__']
        try:
            script_index = sys.argv.index(script)
        except ValueError:
            try:
                script_index = sys.argv.index(os.path.abspath(script))
            except ValueError:
                script_index = 0
    else:
        script_index = 0
    args = sys.argv[script_index+1:script_index+1+n_args]
    if len(args) != n_args:
        usage = usage or "Script requires %d arguments, you supplied %d" % (n_args, len(args))
        raise Exception(usage)
    return args


def get_simulator(*arguments):
    """
    Import and return a PyNN simulator backend module based on command-line
    arguments.
    
    The simulator name should be the first positional argument. If your script
    needs additional arguments, you can specify them as (name, help_text) tuples.
    If you need more complex argument handling, you should use argparse
    directly.
    
    Returns (simulator, command-line arguments)
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("simulator",
                        help="neuron, nest, brian or another backend simulator")
    for argument in arguments:
        arg_name, help_text = argument[:2]
        extra_args = {}
        if len(argument) > 2:
            extra_args = argument[2]
        parser.add_argument(arg_name, help=help_text, **extra_args)
    args = parser.parse_args()
    sim = import_module("pyNN.%s" % args.simulator)
    return sim, args


def init_logging(logfile, debug=False, num_processes=1, rank=0, level=None):
    """
    Simple configuration of logging.
    """
    # allow logfile == None
    # which implies output to stderr
    # num_processes and rank should be obtained using mpi4py, rather than having them as arguments
    if logfile:
        if num_processes > 1:
            logfile += '.%d' % rank
        logfile = os.path.abspath(logfile)

    # prefix log messages with mpi rank
    mpi_prefix = ""
    if num_processes > 1:
        mpi_prefix = 'Rank %d of %d: ' % (rank, num_processes)

    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # allow user to override exact log_level
    if level:
        log_level = level

    logging.basicConfig(level=log_level,
                        format=mpi_prefix+'%(asctime)s %(levelname)-8s [%(name)s] %(message)s (%(pathname)s[%(lineno)d]:%(funcName)s)',
                        filename=logfile,
                        filemode='w')
    return logging.getLogger("PyNN")


def save_population(population, filename, variables=[]):
    """
    Saves the spike_times of a  population and the size, structure, labels such
    that one can load it back into a SpikeSourceArray population using the
    load_population() function.
    """
    import shelve
    s = shelve.open(filename)
    s['spike_times'] = population.getSpikes()
    s['label'] = population.label
    s['size'] = population.size
    s['structure'] = population.structure # should perhaps just save the positions?
    variables_dict = {}
    for variable in variables:
        variables_dict[variable] = getattr(population, variable)
    s['variables'] = variables_dict
    s.close()


def load_population(filename, sim):
    """
    Loads a population that was saved with the save_population() function into
    SpikeSourceArray.
    """
    import shelve
    s = shelve.open(filename)
    ssa = getattr(sim, "SpikeSourceArray")
    population = getattr(sim, "Population")(s['size'], ssa,
                                            structure=s['structure'],
                                            label=s['label'])
    # set the spiketimes
    spikes = s['spike_times']
    for neuron in range(s['size']):
        spike_times = spikes[spikes[:,0] == neuron][:,1]
        neuron_in_new_population = neuron+population.first_id
        index = population.id_to_index(neuron_in_new_population)
        population[index].set_parameters(**{'spike_times':spike_times})
    # set the variables
    for variable, value in s['variables'].items():
        setattr(population, variable, value)
    s.close()
    return population


def normalized_filename(root, basename, extension, simulator, num_processes=None):
    """
    Generate a file path containing a timestamp and information about the
    simulator used and the number of MPI processes.
    
    The date is used as a sub-directory name, the date & time are included in the
    filename.
    """
    timestamp = datetime.now()
    if num_processes:
        np = "_np%d" % num_processes
    else:
        np = ""
    return os.path.join(root,
                        timestamp.strftime("%Y%m%d"),
                        "%s_%s%s_%s.%s" % (basename,
                                           simulator,
                                           np,
                                           timestamp.strftime("%Y%m%d-%H%M%S"),
                                           extension))


def connection_plot(projection, positive='O', zero='.', empty=' ', spacer=''):
    """ """
    connection_array = projection.get('weight', format='array')
    image = numpy.zeros_like(connection_array, dtype=unicode)
    old_settings = numpy.seterr(invalid='ignore')  # ignore the complaint that x > 0 is invalid for NaN
    image[connection_array > 0] = positive
    image[connection_array == 0] = zero
    numpy.seterr(**old_settings)  # restore original floating point error settings
    image[numpy.isnan(connection_array)] = empty
    return '\n'.join([spacer.join(row) for row in image])


class Timer(object):
    """
    For timing script execution.

    Timing starts on creation of the timer.
    """

    def __init__(self):
        self.start()
        self.marks = []

    def start(self):
        """Start/restart timing."""
        self._start_time = time.time()
        self._last_check = self._start_time

    def elapsed_time(self, format=None):
        """
        Return the elapsed time in seconds but keep the clock running.

        If called with ``format="long"``, return a text representation of the
        time. Examples::

            >>> timer.elapsed_time()
            987
            >>> timer.elapsed_time(format='long')
            16 minutes, 27 seconds
        """
        current_time = time.time()
        elapsed_time = current_time - self._start_time
        if format == 'long':
            elapsed_time = Timer.time_in_words(elapsed_time)
        self._last_check = current_time
        return elapsed_time

    @deprecated('elapsed_time()')
    def elapsedTime(self, format=None):
        return self.elapsed_time(format)

    def reset(self):
        """Reset the time to zero, and start the clock."""
        self.start()

    def diff(self, format=None): # I think delta() would be a better name for this method.
        """
        Return the time since the last time :meth:`elapsed_time()` or
        :meth:`diff()` was called.

        If called with ``format='long'``, return a text representation of the
        time.
        """
        current_time = time.time()
        time_since_last_check = current_time - self._last_check
        self._last_check = current_time
        if format=='long':
            time_since_last_check = Timer.time_in_words(time_since_last_check)
        return time_since_last_check

    @staticmethod
    def time_in_words(s):
        """
        Formats a time in seconds as a string containing the time in days,
        hours, minutes, seconds. Examples::

            >>> Timer.time_in_words(1)
            1 second
            >>> Timer.time_in_words(123)
            2 minutes, 3 seconds
            >>> Timer.time_in_words(24*3600)
            1 day
        """
        # based on http://mail.python.org/pipermail/python-list/2003-January/181442.html
        T = {}
        T['year'], s = divmod(s, 31556952)
        min, T['second'] = divmod(s, 60)
        h, T['minute'] = divmod(min, 60)
        T['day'], T['hour'] = divmod(h, 24)
        def add_units(val, units):
            return "%d %s" % (int(val), units) + (val>1 and 's' or '')
        return ', '.join([add_units(T[part], part)
                          for part in ('year', 'day', 'hour', 'minute', 'second')
                          if T[part]>0])

    def mark(self, label):
        """
        Store the time since the last time since the last time
        :meth:`elapsed_time()`, :meth:`diff()` or :meth:`mark()` was called,
        together with the provided label, in the attribute 'marks'.
        """
        self.marks.append((label, self.diff()))


class ProgressBar(object):
    """
    Create a progress bar in the shell.
    """

    def __init__(self, width=77, char="#", mode="fixed"):
        self.char = char
        self.mode = mode
        if not self.mode in ['fixed', 'dynamic']:
            self.mode = 'fixed'
        self.width = width

    def set_level(self, level):
        """
        Rebuild the bar string based on `level`, which should be a number
        between 0 and 1.
        """
        if level < 0:
            level = 0
        if level > 1:
            level = 1

        # figure the proper number of 'character' make up the bar
        all_full = self.width - 2
        num_hashes = int(round(level * all_full))

        if self.mode == 'dynamic':
            # build a progress bar with self.char (to create a dynamic bar
            # where the percent string moves along with the bar progress.
            bar = self.char * num_hashes
        else:
            # build a progress bar with self.char and spaces (to create a
            # fixed bar (the percent string doesn't move)
            bar = self.char * num_hashes + ' ' * (all_full - num_hashes)
        bar = u'[ %s ] %3.0f%%' % (bar, 100*level)
        print(bar, end=u' \r')
        sys.stdout.flush()

    def __call__(self, level):
        self.set_level(level)


def assert_arrays_equal(a, b):
    import numpy
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
    assert (a.flatten()==b.flatten()).all(), "%s != %s" % (a,b)


def assert_arrays_almost_equal(a, b, threshold):
    import numpy
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert a.shape == b.shape, "%s != %s" % (a,b)
    assert (abs(a - b) < threshold).all(), "max(|a - b|) = %s" % (abs(a - b)).max()


def sort_by_column(a, col):
    # see stackoverflow.com/questions/2828059/
    return a[a[:,col].argsort(),:]


# based on http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
class forgetful_memoize(object):
    """
    Decorator that caches the result from the last time a function was called.
    If the next call uses the same arguments, the cached value is returned, and
    not re-evaluated. If the next call uses different arguments, the cached
    value is overwritten.

    The use case is when the same, heavy-weight function is called repeatedly
    with the same arguments in different places.
    """

    def __init__(self, func):
        self.func = func
        self.cached_args = None
        self.cached_value = None

    def __call__(self, *args):
        import pdb; pdb.set_trace()
        if args == self.cached_args:
            print("using cached value")
            return self.cached_value
        else:
            #print("calculating value")
            value = self.func(*args)
            self.cached_args = args
            self.cached_value = value
            return value

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
