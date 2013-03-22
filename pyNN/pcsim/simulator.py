# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes useable by the common implementation:

Functions:
    reset()

Classes:
    ID
    Recorder
    Connection
    
Attributes:
    state -- a singleton instance of the _State class.
    recorder_list

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import logging
import pypcsim
import types
import numpy
from pyNN import common, errors, standardmodels, core

recorder_list = []
STATE_VARIABLE_MAP = {"v": ("Vinit", 1e-3)}

logger = logging.getLogger("PyNN")

# --- Internal PCSIM functionality -------------------------------------------- 

def is_local(id):
    """Determine whether an object exists on the local MPI node."""
    return pypcsim.SimObject.ID(id).node == net.mpi_rank()


# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        """Initialize the simulator."""
        self.initialized = False
        self.t = 0.0
        self.min_delay = None
        self.max_delay = None
        self.constructRNGSeed = None
        self.simulationRNGSeed = None
    
    @property
    def num_processes(self):
        return net.mpi_size()
    
    @property
    def mpi_rank(self):
        return net.mpi_rank()

    dt = property(fget=lambda self: net.get_dt().in_ms()) #, fset=lambda self,x: net.set_dt(pypcsim.Time.ms(x)))

def reset():
    """Reset the state of the current network to time t = 0."""
    net.reset()
    state.t = 0.0
    
    
# --- For implementation of access to individual neurons' parameters -----------

class ID(long, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        long.__init__(n)
        common.IDMixin.__init__(self)
    
    def _pcsim_cell(self):
        """Return the PCSIM cell with the current ID."""
        global net
        #if self.parent:
        #    pcsim_cell = self.parent.pcsim_population.object(self)
        #else:
        pcsim_cell = net.object(self)
        return pcsim_cell
    
    def get_native_parameters(self):
        """Return a dictionary of parameters for the PCSIM cell model."""
        pcsim_cell = self._pcsim_cell()
        pcsim_parameters = {}
        if self.is_standard_cell:
            parameter_names = [D['translated_name'] for D in self.celltype.translations.values()]
        else:
            parameter_names = [] # for native cells, is there a way to get their list of parameters?
        
        for translated_name in parameter_names:
            if hasattr(self.celltype, 'getterMethods') and translated_name in self.celltype.getterMethods:
                getterMethod = self.celltype.getterMethods[translated_name]
                pcsim_parameters[translated_name] = getattr(pcsim_cell, getterMethod)()    
            else:
                try:
                    pcsim_parameters[translated_name] = getattr(pcsim_cell, translated_name)
                except AttributeError, e:
                    raise AttributeError("%s. Possible attributes are: %s" % (e, dir(pcsim_cell)))
        for k,v in pcsim_parameters.items():
            if isinstance(v, pypcsim.StdVectorDouble):
                pcsim_parameters[k] = list(v)
        return pcsim_parameters
    
    def set_native_parameters(self, parameters):
        """Set parameters of the PCSIM cell model from a dictionary."""
        simobj = self._pcsim_cell()
        for name, value in parameters.items():
            if hasattr(self.celltype, 'setterMethods') and name in self.celltype.setterMethods:
                setterMethod = self.celltype.setterMethods[name]
                getattr(simobj, setterMethod)(value)
            else:               
                setattr(simobj, name, value)

    def get_initial_value(self, variable):
        pcsim_name, unit_conversion = STATE_VARIABLE_MAP[variable]
        pcsim_cell = self._pcsim_cell()
        if hasattr(self.celltype, 'getterMethods') and variable in self.celltype.getterMethods:
            getterMethod = self.celltype.getterMethods[pcsim_name]
            value = getattr(pcsim_cell, getterMethod)()    
        else:
            try:
                value = getattr(pcsim_cell, pcsim_name)
            except AttributeError, e:
                raise AttributeError("%s. Possible attributes are: %s" % (e, dir(pcsim_cell)))
        return value/unit_conversion

    def set_initial_value(self, variable, value):
        pcsim_name, unit_conversion = STATE_VARIABLE_MAP[variable]
        pcsim_cell = self._pcsim_cell()
        value = unit_conversion*value
        if hasattr(self.celltype, 'setterMethods') and variable in self.celltype.setterMethods:
            setterMethod = self.celltype.setterMethods[pcsim_name]
            getattr(pcsim_cell, setterMethod)(value)    
        else:
            try:
                value = setattr(pcsim_cell, pcsim_name, value)
            except AttributeError, e:
                raise AttributeError("%s. Possible attributes are: %s" % (e, dir(pcsim_cell)))
        index = self.parent.id_to_local_index(self)
        self.parent.initial_values[variable][index] = value

# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Store an individual connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """
    
    def __init__(self, source, target, pcsim_connection, weight_unit_factor):
        """
        Create a new connection.
        
        `source` -- ID of pre-synaptic neuron.
        `target` -- ID of post-synaptic neuron.
        `pcsim_connection` -- a PCSIM Connection object.
        `weight_unit_factor` -- 1e9 for current-based synapses (A-->nA), 1e6 for
                                conductance-based synapses (S-->µS).
        """
        self.source = source
        self.target = target
        self.pcsim_connection = pcsim_connection
        self.weight_unit_factor = weight_unit_factor
        
    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return self.weight_unit_factor*self.pcsim_connection.W
    def _set_weight(self, w):
        self.pcsim_connection.W = w/self.weight_unit_factor
    weight = property(fget=_get_weight, fset=_set_weight)
    
    def _get_delay(self):
        """Synaptic delay in ms."""
        return 1000.0*self.pcsim_connection.delay # s --> ms
    def _set_delay(self, d):
        self.pcsim_connection.delay = 0.001*d
    delay = property(fget=_get_delay, fset=_set_delay)
    

# --- Initialization, and module attributes ------------------------------------
          
net = None
state = _State()
del _State
