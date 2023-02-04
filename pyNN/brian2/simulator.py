# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the Brian2 simulator.

Classes and attributes usable by the common implementation:

Classes:
    ID

Attributes:
    state -- an instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import brian2
import numpy as np
from .. import common


name = "Brian2"
logger = logging.getLogger("PyNN")

ms = brian2.ms


class ID(int, common.IDMixin):

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


class State(common.control.BaseState):

    def __init__(self):
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self._min_delay = 'auto'
        self.network = None
        self.clear()

    def run(self, simtime):
        for recorder in self.recorders:
            recorder._finalize()
        if not self.running:
            assert self.network.clock.t == 0 * ms
            self.network.store("before-first-run")
            # todo: handle the situation where new Populations or Projections are
            #       created after the first run and then "reset" is called
        self.running = True
        self.network.run(simtime * ms)

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def clear(self):
        self.recorders = set([])
        self.id_counter = 0
        self.current_sources = []
        self.segment_counter = -1
        if self.network:
            for item in self.network.sorted_objects:
                del item
            del self.network
        self.network = brian2.Network()
        self.network.clock = brian2.Clock(0.1 * ms)
        self.running = False
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        if self.running:
            self.network.restore("before-first-run")
        self.running = False
        self.t_start = 0
        self.segment_counter += 1

    def _get_dt(self):
        if self.network.clock is None:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        return float(self.network.clock.dt / ms)

    def _set_dt(self, timestep):
        logger.debug("Setting timestep to %s", timestep)
        # if self.network.clock is None or timestep != self._get_dt():
        #    self.network.clock = brian2.Clock(dt=timestep*ms)
        self.network.clock.dt = timestep * ms
    dt = property(fget=_get_dt, fset=_set_dt)

    @property
    def t(self):
        return float(self.network.clock.t / ms)

    def _get_min_delay(self):
        if self._min_delay == 'auto':
            min_delay = np.inf
            for item in self.network.sorted_objects:
                if isinstance(item, brian2.Synapses):
                    matrix = np.asarray(item.delay) * 10000
                    min_delay = min(min_delay, matrix.min())
            if np.isinf(min_delay):
                self._min_delay = self.dt
            else:
                # Synapses.delay is an integer, the number of time steps
                self._min_delay = min_delay * self.dt
        return self._min_delay

    def _set_min_delay(self, delay):
        self._min_delay = delay
    min_delay = property(fget=_get_min_delay, fset=_set_min_delay)


state = State()
