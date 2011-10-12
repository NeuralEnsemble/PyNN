# encoding: utf8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the MOOSE simulator.

Functions and classes useable by the common implementation:
    run()


All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id:$
"""

import moose
from pyNN import common, core

# global variables
recorder_list = []

ms = 1e-3
in_ms = 1.0/ms

# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.ctx = moose.PyMooseBase.getContext()
        self.gid_counter = 0
        self.num_processes = 1 # we're not supporting MPI
        self.mpi_rank = 0      # for now
        self.min_delay = 0.0
        self.max_delay = 1e12

    @property
    def t(self):
        return self.ctx.getCurrentTime()*in_ms
    
    def __get_dt(self):
        return self.ctx.getClocks()[0]*in_ms
    def __set_dt(self, dt):
        print "setting dt to %g ms" % dt
        self.ctx.setClock(0, dt*ms, 0) # integration clock
        self.ctx.setClock(1, dt*ms, 1) # ?
        self.ctx.setClock(2, dt*ms, 0) # recording clock
    dt = property(fget=__get_dt, fset=__set_dt)

def run(simtime):
    print "simulating for %g ms" % simtime
    state.ctx.reset()
    state.ctx.step(simtime*ms)

def reset():
    state.ctx.reset()

# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__
    
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def _build_cell(self, cell_model, cell_parameters):
        """
        Create a cell in MOOSE, and register its global ID.
        
        `cell_model` -- one of the cell classes defined in the
                        `moose.cells` module (more generally, any class that
                        implements a certain interface, but I haven't
                        explicitly described that yet).
        `cell_parameters` -- a dictionary containing the parameters used to
                             initialise the cell model.
        """
        id = int(self)
        self._cell = cell_model("neuron%d" % id, **cell_parameters)          # create the cell object
        
    def get_native_parameters(self):
        """Return a dictionary of parameters for the NEURON cell model."""
        D = {}
        for name in self._cell.parameter_names:
            D[name] = getattr(self._cell, name)
        return D
    
    def set_native_parameters(self, parameters):
        """Set parameters of the NEURON cell model from a dictionary."""
        for name, val in parameters.items():
            setattr(self._cell, name, val)

state = _State()  # a Singleton, so only a single instance ever exists
del _State