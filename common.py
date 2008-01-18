# encoding: utf-8
"""
Defines the PyNN classes and functions, and hence the FACETS API.
The simulator-specific classes should inherit from these and have the same
arguments.
$Id$
"""
__version__ = "$Revision$"

import types, time, copy, sys
import numpy
from math import *
from pyNN import random

class InvalidParameterValueError(Exception): pass
class NonExistentParameterError(Exception): pass
class InvalidDimensionsError(Exception): pass
class ConnectionError(Exception): pass

dt = 0.1
_min_delay = 0.1

# ==============================================================================
#   Utility classes and functions
# ==============================================================================

# The following two functions taken from
# http://www.nedbatchelder.com/text/pythonic-interfaces.html
def _functionId(obj, nFramesUp):
    """ Create a string naming the function n frames up on the stack. """
    fr = sys._getframe(nFramesUp+1)
    co = fr.f_code
    return "%s.%s" % (obj.__class__, co.co_name)

def _abstractMethod(obj=None):
    """ Use this instead of 'pass' for the body of abstract methods. """
    raise Exception("Unimplemented abstract method: %s" % _functionId(obj, 1))

def build_translations(*translation_list):
    translations = {}
    for item in translation_list:
        pynn_name = item[0]
        sim_name = item[1]
        if len(item) == 2: # no transformation
            f = pynn_name
            g = sim_name
        elif len(item) == 3: # simple multiplicative factor
            scale_factor = item[2]
            f = "float(%g)*%s" % (scale_factor, pynn_name)
            g = "%s/float(%g)" % (sim_name, scale_factor)
        elif len(item) == 4: # more complex transformation
            f = item[2]
            g = item[3]
        translations[pynn_name] = {'translated_name': sim_name,
                                   'forward_transform': f,
                                   'reverse_transform': g}
    return translations


class ID(int):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    non_parameter_attributes = ('parent','_cellclass','cellclass','_position','position','hocname')
    
    def __init__(self,n):
        int.__init__(n)
        self.parent = None
        self._cellclass = None

    def __getattr__(self,name):
        """Note that this currently does not translate units."""
        return _abstractMethod(self)
    
    def __setattr__(self,name,value):
        if name in ID.non_parameter_attributes:
            object.__setattr__(self,name,value)
        else:
            return self.setParameters(**{name:value})

    def _set_cellclass(self, cellclass):
        if self.parent is not None:
            raise Exception("Cell class is determined by the Population and cannot be changed for individual neurons.")
        else:
            self._cellclass = cellclass # should check it is a standard cell class or a string

    def _get_cellclass(self):
        if self.parent is not None:
            celltype = self.parent.celltype
            if isinstance(celltype, str):
                return celltype
            else:
                return celltype.__class__
        else:
            return self._cellclass
        
    cellclass = property(_get_cellclass, _set_cellclass)
    
    def _set_position(self,pos):
        assert isinstance(pos, tuple) or isinstance(pos, numpy.ndarray)
        assert len(pos) == 3
        if self.parent:
            index = numpy.where(self.parent.cell.flatten() == int(self))[0][0]
            self.parent.positions[:,index] = pos
        else:
            self._position = pos
        
    def _get_position(self):
        if self.parent:
            index = numpy.where(self.parent.cell.flatten() == int(self))[0][0]
            return self.parent.positions[:,index]  
        else:
            try:
                return self._position
            except (AttributeError, KeyError):
                self._position = (float(self), 0.0, 0.0)
                return self._position

    position = property(_get_position, _set_position)

    def setParameters(self,**parameters):
        """Set cell parameters, given as a sequence of parameter=value arguments."""
        return _abstractMethod(self)
    
    def getParameters(self):
        """Return a dict of all cell parameters."""
        return _abstractMethod(self)

def distance(src, tgt, mask=None, scale_factor=1.0, offset=0., periodic_boundaries=None): # may need to add an offset parameter
    """
    Return the Euclidian distance between two cells.
    `mask` allows only certain dimensions to be considered, e.g.::
      * to ignore the z-dimension, use `mask=array([0,1])`
      * to ignore y, `mask=array([0,2])`
      * to just consider z-distance, `mask=array([2])`
    `scale_factor` allows for different units in the pre- and post- position
    (the post-synaptic position is multipied by this quantity).
    """
    d = src.position - scale_factor*(tgt.position + offset)
    
    if not periodic_boundaries == None:
        d = numpy.array(map(min,((x_i,y_i) for (x_i,y_i) in zip(abs(d),periodic_boundaries-abs(d)))))
    if mask is not None:
        d = d[mask]
    return numpy.sqrt(numpy.dot(d,d))


def distances(pre, post, mask=None, scale_factor=1.0, offset=0., periodic_boundaries=None):
    """Calculate the entire distance matrix at once.
       From http://projects.scipy.org/pipermail/numpy-discussion/2007-April/027203.html"""
    if isinstance(pre, Population): x = pre.positions
    else: 
        x = pre.position
        x = x.reshape(3,1)
    if isinstance(post, Population): y = post.positions
    else: 
        y = post.position
        y = y.reshape(3,1)
    y = scale_factor*(y + offset)
    d = numpy.zeros((x.shape[1],y.shape[1]), dtype=x.dtype)
    for i in xrange(x.shape[0]):
        diff2 = abs(x[i,:,None] - y[i,:])
        if not periodic_boundaries == None:
            dims  = diff2.shape
            diff2 = diff2.flatten()
            diff2 = numpy.array(map(min,((x_i,y_i) for (x_i,y_i) in zip(diff2,periodic_boundaries[i]-diff2))))
            diff2 = diff2.reshape(dims)
        diff2 **= 2
        d += diff2
    numpy.sqrt(d,d)
    return d




# ==============================================================================
#   Standard cells
# ==============================================================================

class StandardCellType(object):
    """Base class for standardized cell model classes."""
    
    translations = {}
    default_parameters = {}
    
    def checkParameters(self, supplied_parameters, with_defaults=False):
        """Checks that the parameters exist and have values of the correct type."""
        default_parameters = self.__class__.default_parameters
        if with_defaults:
            parameters = copy.copy(default_parameters)
        else:
            parameters = {}
        if supplied_parameters:
            for k in supplied_parameters.keys():
                if default_parameters.has_key(k):
                    if type(supplied_parameters[k]) == type(default_parameters[k]): # same type
                        parameters[k] = supplied_parameters[k]
                    elif type(default_parameters[k]) == types.FloatType: # float and something that can be converted to a float
                        try:
                            parameters[k] = float(supplied_parameters[k]) 
                        except (ValueError, TypeError):
                            raise InvalidParameterValueError, (type(supplied_parameters[k]), type(default_parameters[k]))
                    elif type(default_parameters[k]) == types.ListType: # list and something that can be transformed to a list
                        try:
                            parameters[k] = list(supplied_parameters[k])
                        except TypeError:
                            raise InvalidParameterValueError, (type(supplied_parameters[k]), type(default_parameters[k]))
                    else:
                        raise InvalidParameterValueError, (type(supplied_parameters[k]), type(default_parameters[k]))
                else:
                    raise NonExistentParameterError(k)
        return parameters

    def translate(self,parameters):
        """Translate standardized model names to simulator specific names."""
        parameters = self.checkParameters(parameters)
        translated_parameters = {}
        for k in parameters.keys():
            pname = self.__class__.translations[k][0]
            pval = eval(self.__class__.translations[k][1])
            translated_parameters[pname] = pval
        return translated_parameters
    
    def translate1(self, parameters):
        """Translate standardized model names to simulator specific names.
           Alternative implementation."""
        parameters = self.checkParameters(parameters)
        translated_parameters = {}
        for k in parameters.keys():
            pname = self.__class__.translations[k]['translated_name']
            pval = eval(self.__class__.translations[k]['forward_transform'], globals(), parameters)
            translated_parameters[pname] = pval
        return translated_parameters

    def __init__(self,parameters):
        self.parameters = self.checkParameters(parameters, with_defaults=True)
    
    
class IF_curr_alpha(StandardCellType):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    default_parameters = {
        'v_rest'     : -65.0,   # Resting membrane potential in mV. 
        'cm'         :   1.0,   # Capacity of the membrane in nF
        'tau_m'      :  20.0,   # Membrane time constant in ms.
        'tau_refrac' :   0.0,   # Duration of refractory period in ms. 
        'tau_syn_E'  :   0.3,   # Rise time of the excitatory synaptic alpha function in ms.
        'tau_syn_I'  :   0.5,   # Rise time of the inhibitory synaptic alpha function in ms.
        'i_offset'   :   0.0,   # Offset current in nA
        'v_reset'    : -65.0,   # Reset potential after a spike in mV.
        'v_thresh'   : -50.0,   # Spike threshold in mV.
        'v_init'     : -65.0,   # Membrane potential in mV at t = 0
    }

class IF_curr_exp(StandardCellType):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses"""
    
    default_parameters = {
        'v_rest'     : -65.0,   # Resting membrane potential in mV. 
        'cm'         : 1.0,     # Capacity of the membrane in nF
        'tau_m'      : 20.0,    # Membrane time constant in ms.
        'tau_refrac' : 0.0,     # Duration of refractory period in ms. 
        'tau_syn_E'  : 5.0,     # Decay time of excitatory synaptic current in ms.
        'tau_syn_I'  : 5.0,     # Decay time of inhibitory synaptic current in ms.
        'i_offset'   : 0.0,     # Offset current in nA
        'v_reset'    : -65.0,   # Reset potential after a spike in mV.
        'v_thresh'   : -50.0,   # Spike threshold in mV.
        'v_init'     : -65.0,   # Membrane potential in mV at t = 0
    }

class IF_cond_alpha(StandardCellType):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    default_parameters = {
        'v_rest'     : -65.0,   # Resting membrane potential in mV. 
        'cm'         : 1.0,     # Capacity of the membrane in nF
        'tau_m'      : 20.0,    # Membrane time constant in ms.
        'tau_refrac' : 0.0,     # Duration of refractory period in ms.
        'tau_syn_E'  : 0.3,     # Rise time of the excitatory synaptic alpha function in ms.
        'tau_syn_I'  : 0.5,     # Rise time of the inhibitory synaptic alpha function in ms.
        'e_rev_E'    : 0.0,     # Reversal potential for excitatory input in mV
        'e_rev_I'    : -70.0,   # Reversal potential for inhibitory input in mV
        'v_thresh'   : -50.0,   # Spike threshold in mV.
        'v_reset'    : -65.0,   # Reset potential after a spike in mV.
        'i_offset'   : 0.0,     # Offset current in nA
        'v_init'     : -65.0,   # Membrane potential in mV at t = 0
    }
    
class IF_cond_exp(StandardCellType):
    """Leaky integrate and fire model with fixed threshold and 
    decaying-exponential post-synaptic conductance."""
    
    default_parameters = {
        'v_rest'     : -65.0,   # Resting membrane potential in mV. 
        'cm'         : 1.0,     # Capacity of the membrane in nF
        'tau_m'      : 20.0,    # Membrane time constant in ms.
        'tau_refrac' : 0.0,     # Duration of refractory period in ms.
        'tau_syn_E'  : 5.0,     # Decay time of the excitatory synaptic conductance in ms.
        'tau_syn_I'  : 5.0,     # Decay time of the inhibitory synaptic conductance in ms.
        'e_rev_E'    : 0.0,     # Reversal potential for excitatory input in mV
        'e_rev_I'    : -70.0,   # Reversal potential for inhibitory input in mV
        'v_thresh'   : -50.0,   # Spike threshold in mV.
        'v_reset'    : -65.0,   # Reset potential after a spike in mV.
        'i_offset'   : 0.0,     # Offset current in nA
        'v_init'     : -65.0,   # Membrane potential in mV at t = 0
    }
    
class IF_facets_hardware1(StandardCellType):
    """Leaky integrate and fire model with conductance-based synapses and fixed 
    threshold as it is resembled by the FACETS Hardware Stage 1. For further 
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    
    default_parameters = {
        'cm'                :    0.2,     # nF  
        'g_leak'            :   20.0,     # nS
        'tau_refrac'        :    1.0,     # ms
        'tau_syn_E'         :   20.0,     # ms
        'tau_syn_I'         :   20.0,     # ms
        'v_reset'           :  -80.0,     # mV
        'e_rev_I'           :  -75.0,     # mV,
        'v_rest'            :  -70.0,     # mV
        'v_thresh'          :  -57.0,     # mV
        'e_rev_E'           :    0.0,     # mV        
    }

class HH_cond_exp(StandardCellType):
    """docstring needed here."""
    
    default_parameters = {
        'gbar_Na'   : 20000.0,
        'gbar_K'    : 6000.0,
        'g_leak'    : 10.0,
        'cm'        : 0.2,
        'v_offset'  : -63.0,
        'e_rev_Na'  : 50.0,
        'e_rev_K'   : -90.0,
        'e_rev_leak': -65.0,
        'e_rev_E'   : 0.0,
        'e_rev_I'   : -80.0,
        'tau_syn_E' : 0.2,
        'tau_syn_I' : 2.0,
        'i_offset'  : 0.0,
        'v_init'    : -65.0,
    }

class AdaptiveExponentialIF_alpha(StandardCellType):
    """adaptive exponential integrate and fire neuron according to 
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
            an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642
    """
    
    default_parameters = {
        'v_init'    : -70.6, #'V_m'        # Initial membrane potential in mV
        'w_init'    : 0.0,   #'w'          # Spike-adaptation current in nA
        'cm'        : 0.281, #'C_m'        # Capacity of the membrane in nF
        'tau_refrac': 0.0,   #'t_ref'      # Duration of refractory period in ms.
        'v_spike'   : 0.0,   #'V_peak'     # Spike detection threshold in mV.
        'v_reset'   : -70.6, #'V_reset'    # Reset value for V_m after a spike. In mV.
        'v_rest'    : -70.6, #'E_L'        # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m'     : 9.3667,#'g_L'        # Membrane time constant in ms (nest:Leak conductance in nS.)
        'i_offset'  : 0.0,   #'I_e'        # Offset current in nA
        'a'         : 4.0,                 # Subthreshold adaptation conductance in nS.
        'b'         : 0.0805,              # Spike-triggered adaptation in nA
        'delta_T'   : 2.0,   # Delta_T     # Slope factor in mV
        'tau_w'     : 144.0, #'tau_w'      # Adaptation time constant in ms
        'v_thresh'  : -50.4, #'V_t'        # Spike initiation threshold in mV (V_th can also be used for compatibility).
        'e_rev_E'   : 0.0,   #'E_ex'       # Excitatory reversal potential in mV.
        'tau_syn_E' : 5.0,   #'tau_ex'     # Rise time of excitatory synaptic conductance in ms (alpha function).
        'e_rev_I'   : -80.0, #'E_in'       # Inhibitory reversal potential in mV.
        'tau_syn_I' : 5.0,   #'tau_in'     # Rise time of the inhibitory synaptic conductance in ms (alpha function).
    }

class SpikeSourcePoisson(StandardCellType):
    """Spike source, generating spikes according to a Poisson process."""

    default_parameters = {
        'rate'     : 0.0,       # Mean spike frequency (Hz)
        'start'    : 0.0,       # Start time (ms)
        'duration' : 1e6        # Duration of spike sequence (ms)
    }  

class SpikeSourceArray(StandardCellType):
    """Spike source generating spikes at the times given in the spike_times array."""
    
    default_parameters = { 'spike_times' : [] } # list or numpy array containing spike times in milliseconds.
           

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    dt = timestep
    pass

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    pass
    
def run(simtime):
    """Run the simulation for simtime ms."""
    pass

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    pass

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass,paramDict=None,n=1):
    """Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    pass

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    pass

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    pass

def record(source,filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    pass

def record_v(source,filename):
    """Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    pass

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population:
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    
    def __init__(self,dims,cellclass,cellparams=None,label=None):
        """
        dims should be a tuple containing the population dimensions, or a single
          integer, for a one-dimensional population.
          e.g., (10,10) will create a two-dimensional population of size 10x10.
        cellclass should either be a standardized cell class (a class inheriting
        from common.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        label is an optional name for the population.
        """
        
        self.dim      = dims
        if isinstance(dims, int): # also allow a single integer, for a 1D population
            #print "Converting integer dims to tuple"
            self.dim = (self.dim,)
        else:
            assert isinstance(dims, tuple), "`dims` must be an integer or a tuple."
        self.label    = label
        self.celltype = cellclass
        self.ndim     = len(self.dim)
        self.cellparams = cellparams
        self.size = self.dim[0]
        for i in range(1,self.ndim):
            self.size *= self.dim[i]
        self.cell = None # to be defined by child, simulator-specific classes
    
    def __getitem__(self,addr):
        """Returns a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        pass
    
    def __iter__(self):
        return _abstractMethod(self)
        
    def addresses(self):
        return _abstractMethod(self)
    
    def ids(self):
        return self.__iter__()
    
    def locate(self):
        return _abstractMethod(self)
    
    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size
    
    def _get_positions(self):
        """
        Try to return self._positions. If it does not exist, create it and then return it
        """
        try:
            return self._positions
        except AttributeError:
            x,y,z = numpy.indices(list(self.dim) + [1]*(3-len(self.dim))).astype(float)
            x = x.flatten(); y = y.flatten(); z = z.flatten()
            self._positions = numpy.array((x,y,z))
            return self._positions

    def _set_positions(self, pos_array):
        assert isinstance(pos_array, numpy.ndarray)
        assert pos_array.shape == (3,self.size)
        self._positions = pos_array.copy() # take a copy in case pos_array is changed later

    positions = property(_get_positions, _set_positions, 'A 3xN array (where N is the number of neurons in the Population) giving the x,y,z coordinates of all the neurons (soma, in the case of non-point models).')
    
    def index(self, n):
        """Return the nth cell in the population."""
        return _abstractMethod(self)
    
    def nearest(self, position):
        """Return the neuron closest to the specified position."""
        # doesn't always work correctly if a position is equidistant between two neurons,
        # i.e. 0.5 should be rounded up, but it isn't always.
        pos = numpy.array([position]*self.positions.shape[1]).transpose()
        dist_arr = (self.positions - pos)**2
        distances = dist_arr.sum(axis=0)
        print distances
        nearest = distances.argmin()
        return self.index(nearest)
            
    def set(self,param,val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        return _abstractMethod(self)

    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        return _abstractMethod(self)
    
    def rset(self,parametername,rand_distr):
        """
        'Random' set. Sets the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        return _abstractMethod(self)
    
    def _call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        return _abstractMethod(self)
    
    def _tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        return _abstractMethod(self)

    def randomInit(self,rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        return _abstractMethod(self)

    def record(self,record_from=None,rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        return _abstractMethod(self)

    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        return _abstractMethod(self)

    def printSpikes(self,filename,gather=True,compatible_output=True):
        """
        Writes spike times to file.
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.
        
        If gather is True, the file will only be created on the master node,
        otherwise, a file will be written on each node.
        """        
        return _abstractMethod(self)
    
    def print_v(self,filename,gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        return _abstractMethod(self)
    
    def meanSpikeCount(self,gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        return _abstractMethod(self)

# ==============================================================================

class Projection:
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 method='allToAll', methodParameters=None,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell signals action potentials
        
        target - string specifying which synapse on the postsynaptic cell to connect to
        If source and/or target are not given, default values are used.
        
        method - string indicating which algorithm to use in determining connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'
        
        methodParameters - dict containing parameters needed by the connection method,
        although we should allow this to be a number or string if there is only
        one parameter.
        
        synapse_dynamics - ...
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within methodParameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        
        self.pre    = presynaptic_population  # } these really        
        self.source = source                  # } should be
        self.post   = postsynaptic_population # } read-only
        self.target = target                  # }
        self.label  = label
        self.rng    = rng
        self.synapse_dynamics = synapse_dynamics
        self.connection = None # access individual connections. To be defined by child, simulator-specific classes
        if label is None:
            if self.pre.label and self.post.label:
                self.label = "%s → %s" % (self.pre.label, self.post.label)
    
    def __len__(self):
        """Return the total number of connections."""
        return self.nconn
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
    
    def _oneToOne(self):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        return _abstractMethod(self)
    
    def _fixedProbability(self,parameters):
        """
        For each pair of pre-post cells, the connection probability is constant.
        """
        allow_self_connections = True
        try:
            p_connect = float(parameters)
        except TypeError:
            p_connect = parameters['p_connect']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
    
    def _distanceDependentProbability(self,parameters):
        """
        For each pair of pre-post cells, the connection probability depends on distance.
        d_expression should be the right-hand side of a valid python expression
        for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
        """
        allow_self_connections = True
        if type(parameters) == types.StringType:
            d_expression = parameters
        else:
            d_expression = parameters['d_expression']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
    
    def _fixedNumberPre(self,parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
        elif type(parameters) == types.DictType:
            if parameters.has_key['n']: # all cells have same number of connections
                n = parameters['n']
            elif parameters.has_key['rng']: # number of connections per cell follows a distribution
                rng = parameters['rng']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        else : # assume parameters is a rng
            rng = parameters
    
    def _fixedNumberPost(self,parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
        elif type(parameters) == types.DictType:
            if parameters.has_key['n']: # all cells have same number of connections
                n = parameters['n']
            elif parameters.has_key['rng']: # number of connections per cell follows a distribution
                rng = parameters['rng']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        else : # assume parameters is a rng
            rng = parameters
    
    def _fromFile(self,parameters):
        """
        Load connections from a file.
        """
        if type(parameters) == types.FileType:
            fileobj = parameters
            # check fileobj is already open for reading
        elif type(parameters) == types.StringType:
            filename = parameters
            # now open the file...
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            pass
        
    def _fromList(self,conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        # Need to implement parameter parsing here...
        return _abstractMethod(self)
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        return _abstractMethod(self)
    
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        return _abstractMethod(self)
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        return _abstractMethod(self)
    
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        return _abstractMethod(self)
    
    def setThreshold(self,threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        # This is a bit tricky, because in NEST the spike threshold is a
        # property of the cell model, whereas in NEURON it is a property of the
        # connection (NetCon).
        return _abstractMethod(self)
    
    
    # --- Methods relating to synaptic plasticity ------------------------------
    
    def setupSTDP(self,stdp_model,parameterDict):
        """Set-up STDP."""
        return _abstractMethod(self)
    
    def toggleSTDP(self,onoff):
        """Turn plasticity on or off."""
        return _abstractMethod(self)
    
    def setMaxWeight(self,wmax):
        """Note that not all STDP models have maximum or minimum weights."""
        return _abstractMethod(self)
    
    def setMinWeight(self,wmin):
        """Note that not all STDP models have maximum or minimum weights."""
        return _abstractMethod(self)
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self,filename,gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        return _abstractMethod(self)
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        return _abstractMethod(self)
    
    def weightHistogram(self,min=None,max=None,nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        return _abstractMethod(self)


# ==============================================================================
#   Connection method classes
# ==============================================================================

class Connector(object):
    """Base class for Connector classes."""
    
    def __init__(self, weights, delays):
        self.w_index = 0 # should probably use a generator
        self.d_index = 0 # rather than storing these values
        self.weights = weights
        self.delays = delays
    
    def connect(self,projection):
        """Connect all neurons in ``projection``"""
        _abstractMethod(self)
        
    def getWeights(self, N):
        """
        Returns the next N weight values
        """
        if isinstance(self.weights, random.RandomDistribution): # random
            weights = numpy.array(self.weights.next(N))
        elif isinstance(self.weights, int) or isinstance(self.weights, float):  # int, float
            weights = numpy.ones((N,))*float(self.weights)
        elif hasattr(self.weights, "__len__"):                                            # numpy array
            weights = self.weights[self.w_index:self.w_index+N]
        else:
            raise Exception("weights is of type %s" % type(self.weights))
        assert numpy.all(weights>=0), "Weight values must be positive"
        self.w_index += N
        return weights
    
    def getDelays(self, N, start=0):
        """
        Returns the next N delays values
        """
        if isinstance(self.delays, random.RandomDistribution): # random
            delays = numpy.array(self.delays.next(N))
        elif isinstance(self.weights, int) or isinstance(self.weights, float):  # int, float
            delays = numpy.ones((N,))*float(self.delays)
        elif hasattr(delays, "__len__"):                           # numpy array
            delays = self.delays[self.d_index:self.d_index+N]
        else:
            raise Exception("delays is of type %s" % type(self.delays))
        assert numpy.all(delays>=_min_delay), "Delay values must be greater than the minimum delay"
        self.d_index += N
        return delays
    
class AllToAllConnector(Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    
    def __init__(self, allow_self_connections=True, weights=0.0, delays=_min_delay):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        
class FixedNumberPostConnector(Connector):
    """
    Each postsynaptic cell receives a fixed number of connections, chosen
    randomly from the presynaptic cells.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=_min_delay):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) > 0), "the random distribution produces negative numbers"
        else:
            raise Exception("n must be an integer or a RandomDistribution object")

class FixedNumberPreConnector(Connector):
    """
    Connects all cells in the postsynaptic population to fixed number of
    cells in the presynaptic population, randomly choosen.
    """
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=_min_delay):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) > 0), "the random distribution produces negative numbers"
        else:
            raise Exception("n must be an integer or a RandomDistribution object")

class OneToOneConnector(Connector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell i in the presynaptic population to cell i in the postsynaptic
    population for all i.
    In fact, despite the name, this should probably be generalised to the
    case where the pre and post populations have different dimensions, e.g.,
    cell i in a 1D pre population of size n should connect to all cells
    in row i of a 2D post population of size (n,m).
    """
    
    def __init__(self, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
    
class FixedProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability is constant.
    """
    
    def __init__(self, p_connect, allow_self_connections=True, weights=0.0, delays=_min_delay):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        assert 0 <= self.p_connect
        
class DistanceDependentProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    d_expression should be the right-hand side of a valid python expression
    for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
    If axes is not supplied, then the 3D distance is calculated. If supplied,
    axes should be a string containing the axes to be used, e.g. 'x', or 'yz'
    axes='xyz' is the same as axes=None.
    It may be that the pre and post populations use different units for position, e.g.
    degrees and µm. In this case, `scale_factor` can be specified, which is applied
    to the positions in the post-synaptic population. An offset can also be included.
    """
    
    AXES = {'x' : [0],    'y': [1],    'z': [2],
            'xy': [0,1], 'yz': [1,2], 'xz': [0,2], 'xyz': None, None: None}
    
    def __init__(self, d_expression, axes=None, scale_factor=1.0, offset=0.,
                 periodic_boundaries=False, allow_self_connections=True,
                 weights=0.0, delays=_min_delay):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        assert isinstance(d_expression, str)
        try:
            d = 0; assert 0 <= eval(d_expression), eval(d_expression)
            d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError:
            print d_expression
            raise
        self.d_expression = d_expression
        self.allow_self_connections = allow_self_connections
        self.mask = DistanceDependentProbabilityConnector.AXES[axes]
        self.periodic_boundaries = periodic_boundaries
        if self.mask is not None:
            self.mask = numpy.array(self.mask)
        self.scale_factor = scale_factor
        self.offset = offset
        

        
# ==============================================================================
#   Synapse Dynamics classes
# ==============================================================================

class SynapseDynamics(object):
    """
    For specifying synapse short-term (faciliation,depression) and long-term
    (STDP) plasticity. To be passed as the `synapse_dynamics` argument to
    `Projection.__init__()` or `connect()`.
    """
    
    def __init__(self, fast=None, slow=None):
        self.fast = fast
        self.slow = slow
                
class ShortTermPlasticityMechanism(object):
    """Abstract base class for models of short-term synaptic dynamics."""
    # implement a translation mechanism here, as for StandardCell ?
    
    def __init__(self):
        _abstractMethod(self)

class STDPMechanism(object):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None):
        self.timing_dependence = timing_dependence
        self.weight_dependence = weight_dependence
        self.voltage_dependence = voltage_dependence

class TsodkysMarkramMechanism(ShortTermPlasticityMechanism):
    
    def __init__(self, U, D, F, u0, r0, f0):
        self.U = U # use parameter
        self.D = D # depression time constant (ms)
        self.F = F # facilitation time constant (ms)
        self.u0 = u0 # } initial 
        self.r0 = r0 # } values
        
class STDPWeightDependence(object):
    """Abstract base class for models of STDP weight dependence."""
    
    def __init__(self):
        _abstractMethod(self)
        
class STDPTimingDependence(object):
    """Abstract base class for models of STDP timing dependence (triplets, etc)"""
    
    def __init__(self):
        _abstractMethod(self)

class AdditiveWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.
    """
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        self.w_min = w_min
        self.w_max = w_max
        self.A_plus = A_plus
        self.A_minus = A_minus

class MultiplicativeWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        pass
    

class AdditivePotentiationMultiplicativeDepression(STDPWeightDependence):
    """
    For depression, Dw propto w-w_min
    For potentiation, Dw constant
    (van Rossum rule?)
    """

    def __init__(self, w_min=0.0, A_plus=0.01, A_minus=0.01):
        pass
    
class GutigWeightDependence(STDPWeightDependence):
    pass

class PfisterSpikeTripletRule(STDPTimingDependence):
    pass

class SpikePairRule(STDPTimingDependence):
    
    def __init__(self, tau_plus, tau_minus):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus


# ==============================================================================
#   Utility classes
# ==============================================================================
   
class Timer:
    """For timing script execution."""
    # Note that this class only has static methods, i.e. there is only one timer.
    # It might be nice to allow instances to be created, i.e. have the possibility
    # of multiple independent timers.
    
    @staticmethod
    def start():
        """Start timing."""
        global start_time
        start_time = time.time()
    
    @staticmethod
    def elapsedTime():
        """Return the elapsed time but keep the clock running."""
        return time.time() - start_time
    
    @staticmethod
    def reset():
        """Reset the time to zero, and start the clock."""
        global start_time
        start_time = time.time()
    
# ==============================================================================
