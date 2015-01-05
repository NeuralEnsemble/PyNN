"""
Current source classes for the pcsim module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import pypcsim
from pyNN.pcsim import simulator


class CurrentSource(object):
    """Base class for a source of current to be injected into a neuron."""

    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        if simulator.state.num_processes == 1:
            delay = 0.0
        else:
            delay = simulator.state.min_delay # perhaps it would be better to replicate the current source on each node, to avoid this delay
        for cell in cell_list:
            if cell.local and not cell.celltype.injectable:
                raise TypeError("Can't inject current into a spike source.")
            c = simulator.net.connect(self.input_node, cell, pypcsim.StaticAnalogSynapse(delay=0.001*delay))
            self.connections.append(c)


class StepCurrentSource(CurrentSource):
    """A step-wise time-varying current source."""

    def __init__(self, times, amplitudes):
        """Construct the current source.

        Arguments:
            times      -- list/array of times at which the injected current changes.
            amplitudes -- list/array of current amplitudes to be injected at the
                          times specified in `times`.

        The injected current will be zero up until the first time in `times`. The
        current will continue at the final value in `amplitudes` until the end
        of the simulation.
        """
        CurrentSource.__init__(self)
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        self.times = times
        self.amplitudes = amplitudes
        n = len(times)
        durations = numpy.empty((n+1,))
        levels = numpy.empty_like(durations)
        durations[0] = times[0]
        levels[0] = 0.0
        t = numpy.array(times)
        try:
            durations[1:-1] = t[1:] - t[0:-1]
        except ValueError as e:
            raise ValueError("%s. durations[1:].shape=%s, t[1:].shape=%s, t[0:-1].shape=%s" % (e, durations[1:].shape, t[1:].shape, t[0:-1].shape))
        levels[1:] = amplitudes[:]
        durations[-1] = 1e12
        levels *= 1e-9    # nA --> A
        durations *= 1e-3 # s --> ms
        self.input_node = simulator.net.create(pypcsim.AnalogLevelBasedInputNeuron(levels, durations))
        self.connections = []
        print ("created stepcurrentsource")

class DCSource(StepCurrentSource):
    """Source producing a single pulse of current of constant amplitude."""

    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """Construct the current source.

        Arguments:
            start     -- onset time of pulse in ms
            stop      -- end of pulse in ms
            amplitude -- pulse amplitude in nA
        """
        times = [0.0, start, (stop or 1e99)]
        amplitudes = [0.0, amplitude, 0.0]
        StepCurrentSource.__init__(self, times, amplitudes)

