"""
A collection of utility functions and classes.

Functions:
    colour()          - allows output of different coloured text on stdout.
    notify()          - send an e-mail when a simulation has finished.
    get_script_args() - get the command line arguments to the script, however
                        it was run (python, nrniv, mpirun, etc.).
    init_logging()    - convenience function for setting up logging to file and
                        to the screen.
    
Classes:
    MultiSim - a small framework to make it easier to compare results between
               different simulators.
    Timer    - a convenience wrapper around the time.time() function from the
               standard library.

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

red     = 0010; green  = 0020; yellow = 0030; blue = 0040;
magenta = 0050; cyan   = 0060; bright = 0100
try:
    import ll.ansistyle
    def colour(col, text):
        return str(ll.ansistyle.Text(col,str(text)))
except ImportError:
    def colour(col, text):
        return text


def notify(msg="Simulation finished.", subject="Simulation finished.", smtphost=SMTPHOST, address=EMAIL):
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

def get_script_args(script, n_args, usage=''):
    try:
        script_index = sys.argv.index(script)
    except ValueError:
        try:
            script_index = sys.argv.index(os.path.abspath(script))
        except ValueError:
            script_index = 0 # ipython sets __file__ to 'IPython/FakeModule.pyc'
    args = sys.argv[script_index+1:script_index+1+n_args]
    if len(args) != n_args:
        usage = usage or "Script requires %d arguments, you supplied %d" % (n_args, len(args))
        raise Exception(usage)
    return args
    
def init_logging(logfile, debug=False, num_processes=1, rank=0):
    if num_processes > 1:
        logfile += '.%d' % rank
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=logfile,
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=logfile,
                    filemode='w')
        
        
class MultiSim(object):
    """
    A small framework to make it easier to run the same model on multiple
    simulators.
    
    Currently runs the simulations interleaved, but it would be nice to add
    parallel runs via threading, multiple processes, or MPI processes.
    """
    
    def __init__(self, sim_list, net, parameters):
        """
        Build the model defined in the class `net`, with parameters `parameters`,
        for each of the simulator modules specified in `sim_list`.
        
        The `net` constructor takes arguments `sim` and `parameters`.
        """
        self.sim_list = sim_list
        self.nets = {}
        for sim in sim_list:
            self.nets[sim.__name__] = net(sim, parameters)
            
    def __iter__(self):
        return self.nets.itervalues()
    
    def __getattr__(self, name):
        """
        Assumes `name` is a method of the `net` model.
        Return a function that runs `net.name()` for all the simulators.
        """
        def iterate_over_nets(*args, **kwargs):
            retvals = {}
            for sim_name, net in self.nets.items():
                retvals[sim_name] = getattr(net, name)(*args, **kwargs)
            return retvals
        return iterate_over_nets
            
    def run(self, simtime, steps=1, *callbacks):
        """
        Run the model for a time `simtime` in all simulators.
        
        The run may be broken into a number of steps (each of equal duration).
        Any functions in `callbacks` will be called after each step.
        """
        dt = float(simtime)/steps
        for i in range(steps):
            for sim in self.sim_list:
                sim.run(dt)
            for func in callbacks:
                func()
                
    def end(self):
        for sim in self.sim_list:
            sim.end()
            
            
def save_population(population,filename,variables=[]):
    """
    Saves the spike_times of a  population and the dim, size, labels such that one can load it back into a SpikeSourceArray population using the load_population function.
    """
    import shelve
    s = shelve.open(filename)
    s['spike_times'] = population.getSpikes()
    s['label'] = population.label
    s['dim'] = population.dim
    s['size'] = population.size
    variables_dict = {}
    for variable in variables:
        variables_dict[variable] = eval('population.%s'%variable)
    s['variables'] = variables_dict
    s.close()

def load_population(filename):
    """
    Loads a population that was saved with the save_population function into SpikeSourceArray.
    """
    import shelve
    s = shelve.open(filename)
    population = Population(s['dim'],SpikeSourceArray,label=s['label'])
    # set the spiketimes
    spikes = s['spike_times']
    for neuron in numpy.arange(s['size']):
        spike_times = spikes[spikes[:,0]==neuron][:,1]
        neuron_in_new_population = neuron+population.first_id
        index = population.locate(neuron_in_new_population)
        population[index].set_parameters(**{'spike_times':spike_times})
    # set the variables
    for variable, value in s['variables'].items():
        exec('population.%s = value'%variable)
    s.close()
    return population

class Timer(object):
    """For timing script execution."""
    
    def __init__(self):
        self.start()
    
    def start(self):
        """Start timing."""
        self._start_time = time.time()
    
    def elapsedTime(self, format=None):
        """Return the elapsed time in seconds but keep the clock running."""
        elapsed_time = time.time() - self._start_time
        if format=='long':
            elapsed_time = Timer.time_in_words(elapsed_time)
        return elapsed_time
    
    def reset(self):
        """Reset the time to zero, and start the clock."""
        self.start()
    
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
        return ', '.join([add_units(T[part], part) for part in ('year', 'day', 'hour', 'minute', 'second') if T[part]>0])
