# encoding: utf8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the MOOSE simulator.

Functions and classes useable by the common implementation:

Functions:
    create_cells()


All other functions and classes are private, and should not be used by other
modules.
    
$Id:$
"""

import moose
import numpy
from pyNN import common, standardmodels

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


# --- For implementation of create() and Population.__init__() -----------------

def create_cells(cellclass, cellparams, n, parent=None):
    """
    Create cells in MOOSE.
    
    `cellclass`  -- a PyNN standard cell or a native MOOSE cell class that
                   implements an as-yet-undescribed interface.
    `cellparams` -- a dictionary of cell parameters.
    `n`          -- the number of cells to create
    `parent`     -- the parent Population, or None if the cells don't belong to
                    a Population.
    
    This function is used by both `create()` and `Population.__init__()`
    
    Return:
        - a 1D array of all cell IDs
        - a 1D boolean array indicating which IDs are present on the local MPI
          node
        - the ID of the first cell created
        - the ID of the last cell created
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type) and issubclass(cellclass, standardmodels.StandardCellType):
        celltype = cellclass(cellparams)
        cell_model = celltype.model
        cell_parameters = celltype.parameters
    else:
        print cellclass
        raise Exception("Only standard cells currently supported.")
    first_id = state.gid_counter
    last_id = state.gid_counter + n - 1
    all_ids = numpy.array([id for id in range(first_id, last_id+1)], ID)
    # mask_local is used to extract those elements from arrays that apply to the cells on the current node
    mask_local = all_ids%state.num_processes==state.mpi_rank # round-robin distribution of cells between nodes
    for i,(id,is_local) in enumerate(zip(all_ids, mask_local)):
        all_ids[i] = ID(id)
        all_ids[i].parent = parent
        if is_local:
            all_ids[i].local = True
            all_ids[i]._build_cell(cell_model, cell_parameters)
        else:
            all_ids[i].local = False
    state.gid_counter += n
    return all_ids, mask_local, first_id, last_id

state = _State()  # a Singleton, so only a single instance ever exists
del _State