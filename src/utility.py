"""
A collection of utility functions and classes.

Functions:
    colour()          - allows output of different coloured text on stdout.
    notify()          - send an e-mail when a simulation has finished.
    get_script_args() - get the command line arguments to the script, however
                        it was run (python, nrniv, mpirun, etc.).
    init_logging()    - convenience function for setting up logging to file and
                        to the screen.
    
    Timer    - a convenience wrapper around the time.time() function from the
               standard library.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

# If there is a settings.py file on the path, defaults will be
# taken from there.
try:
    from settings import SMTPHOST, EMAIL
except ImportError:
    SMTPHOST = None
    EMAIL = None
import sys
import logging
import time
import os

red     = 0010; green  = 0020; yellow = 0030; blue = 0040
magenta = 0050; cyan   = 0060; bright = 0100
try:
    import ll.ansistyle
    def colour(col, text):
        return str(ll.ansistyle.Text(col, str(text)))
except ImportError:
    def colour(col, text):
        return text


def notify(msg="Simulation finished.", subject="Simulation finished.",
           smtphost=SMTPHOST, address=EMAIL):
    """Send an e-mail stating that the simulation has finished."""
    if not (smtphost and address):
        print "SMTP host and/or e-mail address not specified.\nUnable to send notification message."
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
    
def init_logging(logfile, debug=False, num_processes=1, rank=0, level=None):
    # allow logfile == None
    # which implies output to stderr
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
                        format=mpi_prefix+'%(asctime)s %(levelname)s %(message)s',
                        filename=logfile,
                        filemode='w')


def save_population(population, filename, variables=[]):
    """
    Saves the spike_times of a  population and the size, structure, labels such that one can load it back into a SpikeSourceArray population using the load_population function.
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
    Loads a population that was saved with the save_population function into SpikeSourceArray.
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


class Timer(object):
    """For timing script execution."""
    
    def __init__(self):
        self.start()
    
    def start(self):
        """Start timing."""
        self._start_time = time.time()
        self._last_check = self._start_time
    
    def elapsedTime(self, format=None):
        """Return the elapsed time in seconds but keep the clock running."""
        current_time = time.time()
        elapsed_time = current_time - self._start_time
        if format == 'long':
            elapsed_time = Timer.time_in_words(elapsed_time)
        self._last_check = current_time
        return elapsed_time
    
    def reset(self):
        """Reset the time to zero, and start the clock."""
        self.start()
    
    def diff(self, format=None): # I think delta() would be a better name for this method.
        """Return the time since the last time elapsedTime() or diff() was called."""
        current_time = time.time()
        time_since_last_check = current_time - self._last_check
        self._last_check = current_time
        if format=='long':
            time_since_last_check = Timer.time_in_words(elapsed_time)
        return time_since_last_check
    
    @staticmethod
    def time_in_words(s):
        """Formats a time in seconds as a string containing the time in days,
        hours, minutes, seconds. Examples::
            >>> time_in_words(1)
            1 second
            >>> time_in_words(123)
            2 minutes, 3 seconds
            >>> time_in_words(24*3600)
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


class ProgressBar:
    """
    Create a progress bar in the shell.
    """
    
    def __init__(self, min_value=0, max_value=100, width=77, **kwargs):
        self.char = kwargs.get('char', '#')
        self.mode = kwargs.get('mode', 'dynamic') # fixed or dynamic
        if not self.mode in ['fixed', 'dynamic']:
            self.mode = 'fixed'
 
        self.bar = ''
        self.min = min_value
        self.max = max_value
        self.span = max_value - min_value
        self.width = width
        self.amount = 0       # When amount == max, we are 100% done 
        self.update_amount(0) 
 
 
    def increment_amount(self, add_amount = 1):
        """
        Increment self.amount by 'add_ammount' or default to incrementing
        by 1, and then rebuild the bar string. 
        """
        new_amount = self.amount + add_amount
        if new_amount < self.min: new_amount = self.min
        if new_amount > self.max: new_amount = self.max
        self.amount = new_amount
        self.build_bar()
 
 
    def update_amount(self, new_amount = None):
        """
        Update self.amount with 'new_amount', and then rebuild the bar 
        string.
        """
        if not new_amount: new_amount = self.amount
        if new_amount < self.min: new_amount = self.min
        if new_amount > self.max: new_amount = self.max
        self.amount = new_amount
        self.build_bar()
 
 
    def build_bar(self):
        """
        Figure new percent complete, and rebuild the bar string base on 
        self.amount.
        """
        diff = float(self.amount - self.min)
        try:
            percent_done = int(round((diff / float(self.span)) * 100.0))
        except Exception:
            percent_done = 100
 
        # figure the proper number of 'character' make up the bar 
        all_full = self.width - 2
        num_hashes = int(round((percent_done * all_full) / 100))
 
        if self.mode == 'dynamic':
            # build a progress bar with self.char (to create a dynamic bar
            # where the percent string moves along with the bar progress.
            self.bar = self.char * num_hashes
        else:
            # build a progress bar with self.char and spaces (to create a 
            # fixe bar (the percent string doesn't move)
            self.bar = self.char * num_hashes + ' ' * (all_full-num_hashes)
 
        percent_str = str(percent_done) + "%"
        self.bar = '[ ' + self.bar + ' ] ' + percent_str
 
 
    def __str__(self):
        return str(self.bar)


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