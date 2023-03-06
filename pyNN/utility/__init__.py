# encoding: utf-8
"""
A collection of utility functions and classes.

Functions:
    notify()          - send an e-mail when a simulation has finished.
    get_script_args() - get the command line arguments to the script, however
                        it was run (python, nrniv, mpirun, etc.).
    get_simulator() -
    init_logging()    - convenience function for setting up logging to file and
                        to the screen.
    save_population()
    load_population()
    normalized_filename()
    sort_by_column()
    forgetful_memoize()

    plotting module

    Timer    - a convenience wrapper around the time.perf_counter() function from the
               standard library.
    ProgressBar
    SimulationProgressBar

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import functools

from .progress_bar import ProgressBar, SimulationProgressBar  # noqa: F401
from .script_tools import (                                   # noqa: F401
    get_script_args,
    get_simulator,
    normalized_filename,
    init_logging,
    notify
)
from .timer import Timer                                      # noqa: F401


# todo: review whether it is worth keeping the following, little-used functions


def save_population(population, filename, variables=None):
    """
    Saves the spike_times of a  population and the size, structure, labels such
    that one can load it back into a SpikeSourceArray population using the
    load_population() function.
    """
    import shelve

    s = shelve.open(filename)
    s["spike_times"] = population.getSpikes()
    s["label"] = population.label
    s["size"] = population.size
    s["structure"] = population.structure  # should perhaps just save the positions?
    variables_dict = {}
    if variables:
        for variable in variables:
            variables_dict[variable] = getattr(population, variable)
    s["variables"] = variables_dict
    s.close()


def load_population(filename, sim):
    """
    Loads a population that was saved with the save_population() function into
    SpikeSourceArray.
    """
    import shelve

    s = shelve.open(filename)
    ssa = getattr(sim, "SpikeSourceArray")
    population = getattr(sim, "Population")(
        s["size"], ssa, structure=s["structure"], label=s["label"]
    )
    # set the spiketimes
    spikes = s["spike_times"]
    for neuron in range(s["size"]):
        spike_times = spikes[spikes[:, 0] == neuron][:, 1]
        neuron_in_new_population = neuron + population.first_id
        index = population.id_to_index(neuron_in_new_population)
        population[index].set_parameters(**{"spike_times": spike_times})
    # set the variables
    for variable, value in s["variables"].items():
        setattr(population, variable, value)
    s.close()
    return population


def sort_by_column(a, col):
    # see stackoverflow.com/questions/2828059/
    return a[a[:, col].argsort(), :]


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
        if args == self.cached_args:
            print("using cached value")
            return self.cached_value
        else:
            value = self.func(*args)
            self.cached_args = args
            self.cached_value = value
            return value

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
