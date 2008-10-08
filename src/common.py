# encoding: utf-8
"""
Defines the PyNN classes and functions, and hence the FACETS API.
The simulator-specific classes should inherit from these and have the same
arguments.
$Id$
"""

import types, time, copy, sys
import numpy
from math import *
from pyNN import random
from string import Template

DEFAULT_WEIGHT = 0.0
DEFAULT_BUFFER_SIZE = 10000

class InvalidParameterValueError(Exception): pass

class NonExistentParameterError(Exception):
    """
    Raised when an attempt is made to access a model parameter that does not
    exist.
    """
    
    def __init__(self, parameter_name, model):
        self.parameter_name = parameter_name
        self.model_name = model.__name__
        if issubclass(model, StandardModelType):
            self.valid_parameter_names = model.default_parameters.keys()
            self.valid_parameter_names.sort()
        else:
            self.valid_parameter_names = ['unknown']

    def __str__(self):
        return "%s (valid parameters for %s are: %s)" % (self.parameter_name,
                                                         self.model_name,
                                                         ", ".join(self.valid_parameter_names))

class InvalidDimensionsError(Exception): pass
class ConnectionError(Exception): pass
class InvalidModelError(Exception): pass
class RoundingWarning(Warning): pass


# ==============================================================================
#   Utility classes and functions
# ==============================================================================

# The following two functions taken from
# http://www.nedbatchelder.com/text/pythonic-interfaces.html
def _function_id(obj, n_frames_up):
    """ Create a string naming the function n frames up on the stack. """
    fr = sys._getframe(n_frames_up+1)
    co = fr.f_code
    return "%s.%s" % (obj.__class__, co.co_name)

def _abstract_method(obj=None):
    """ Use this instead of 'pass' for the body of abstract methods. """
    # Note that there is a NotImplementedError built-in exception we could use
    raise Exception("Unimplemented abstract method: %s" % _function_id(obj, 1))

def is_listlike(obj):
    return hasattr(obj, "__len__") and not isinstance(obj, basestring)

def build_translations(*translation_list):
    """
    Build a translation dictionary from a list of translations/transformations.
    """
    translations = {}
    for item in translation_list:
        assert 2 <= len(item) <= 4, "Translation tuples must have between 2 and 4 items"
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


class IDMixin(object):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object.
    """
    # Simulator ID classes should inherit both from the base type of the ID
    # (e.g., int or long) and from IDMixin.
    # Ideally, the base type need not be numeric, but the position property
    # will have to be modified for that (probably break off into another Mixin
    # class
    
    non_parameter_attributes = ('parent', '_cellclass', 'cellclass',
                                '_position', 'position', 'hocname', '_cell',
                                'inject')
    
    def __init__(self):
        self.parent = None
        self._cellclass = None

    def __getattr__(self, name):
        if name in IDMixin.non_parameter_attributes:
            val = self.__getattribute__(name)
        else:
            try:
                val = self.get_parameters()[name]
            except KeyError:
                raise NonExistentParameterError(name, self.cellclass)
        return val
    
    def __setattr__(self, name, value):
        if name in IDMixin.non_parameter_attributes:
            object.__setattr__(self, name, value)
        else:
            return self.set_parameters(**{name:value})

    def set_parameters(self, **parameters):
        """Set cell parameters, given as a sequence of parameter=value arguments."""
        # if some of the parameters are computed from the values of other
        # parameters, need to get and translate all parameters
        if self.is_standard_cell():
            computed_parameters = self.cellclass.computed_parameters()
            have_computed_parameters = numpy.any([p_name in computed_parameters for p_name in parameters])
            if have_computed_parameters:     
                all_parameters = self.get_parameters()
                all_parameters.update(parameters)
                parameters = all_parameters
            parameters = self.cellclass.translate(parameters)
        self.set_native_parameters(parameters)
    
    def get_parameters(self):
        """Return a dict of all cell parameters."""
        parameters  = self.get_native_parameters()
        if self.is_standard_cell():
            parameters = self.cellclass.reverse_translate(parameters)
        return parameters

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
    
    def is_standard_cell(self):
        return (type(self.cellclass) == type and issubclass(self.cellclass, StandardCellType))
        
    def _set_position(self, pos):
        """
        Set the cell position in 3D space.
        
        Cell positions are stored in an array in the parent Population, if any,
        or within the ID object otherwise.
        """
        assert isinstance(pos, (tuple, numpy.ndarray))
        assert len(pos) == 3
        if self.parent:
            # the following line makes too many assumptions about the
            # implementation of Population. Should use a public method of
            # Population.
            index = numpy.where(self.parent.cell.flatten() == int(self))[0][0]
            self.parent.positions[:, index] = pos
        else:
            self._position = pos
        
    def _get_position(self):
        """
        Return the cell position in 3D space.
        
        Cell positions are stored in an array in the parent Population, if any,
        or within the ID object otherwise. Positions are generated the first
        time they are requested and then cached.
        """
        if self.parent:
            index = numpy.where(self.parent.cell.flatten() == int(self))[0][0]
            return self.parent.positions[:, index]  
        else:
            try:
                return self._position
            except (AttributeError, KeyError):
                self._position = (float(self), 0.0, 0.0)
                return self._position

    position = property(_get_position, _set_position)
      
    def inject(self, current_source):
        """Inject current from a current source object into the cell."""
        current_source.inject_into([self])
        

def distance(src, tgt, mask=None, scale_factor=1.0, offset=0.0,
             periodic_boundaries=None): # may need to add an offset parameter
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
        d = numpy.minimum(abs(d),periodic_boundaries-abs(d))
    if mask is not None:
        d = d[mask]
    return numpy.sqrt(numpy.dot(d, d))


def distances(pre, post, mask=None, scale_factor=1.0, offset=0.0,
              periodic_boundaries=None):
    """
    Calculate the entire distance matrix at once.
    From http://projects.scipy.org/pipermail/numpy-discussion/2007-April/027203.html
    """
    # Note that `mask` is not used.
    if isinstance(pre, Population):
        x = pre.positions
    else: 
        x = pre.position
        x = x.reshape(3, 1)
    if isinstance(post, Population):
        y = post.positions
    else: 
        y = post.position
        y = y.reshape(3, 1)
    y = scale_factor*(y + offset)
    d = numpy.zeros((x.shape[1], y.shape[1]), dtype=x.dtype)
    for i in xrange(x.shape[0]):
        diff2 = abs(x[i,:,None] - y[i,:])
        if periodic_boundaries:
            dims  = diff2.shape
            diff2 = diff2.flatten()
            diff2 = numpy.minimum(diff2, periodic_boundaries[i]-diff2)
            diff2 = diff2.reshape(dims)
        diff2 **= 2
        d += diff2
    numpy.sqrt(d, d)
    return d


# ==============================================================================
#   Standard cells
# ==============================================================================

class StandardModelType(object):
    """Base class for standardized cell model and synapse model classes."""
    
    translations = {}
    default_parameters = {}
    
    def __init__(self, parameters):
        self.parameters = self.__class__.checkParameters(parameters, with_defaults=True)
        self.parameters = self.__class__.translate(self.parameters)
    
    @classmethod
    def checkParameters(cls, supplied_parameters, with_defaults=False):
        """
        Returns a parameter dictionary, checking that each
        supplied_parameter is in the default_parameters and
        converts to the type of the latter.

        If with_defaults==True, parameters not in
        supplied_parameters are in the returned dictionary
        as in default_parameters.

        """
        default_parameters = cls.default_parameters
        if with_defaults:
            parameters = copy.copy(default_parameters)
        else:
            parameters = {}
        if supplied_parameters:
            for k in supplied_parameters.keys():
                if default_parameters.has_key(k):
                    err_msg = "For %s in %s, expected %s, got %s (%s)" % \
                              (k, cls.__name__, type(default_parameters[k]),
                               type(supplied_parameters[k]), supplied_parameters[k])
                    # same type
                    if type(supplied_parameters[k]) == type(default_parameters[k]): 
                        parameters[k] = supplied_parameters[k]
                    # float and something that can be converted to a float
                    elif type(default_parameters[k]) == types.FloatType: 
                        try:
                            parameters[k] = float(supplied_parameters[k]) 
                        except (ValueError, TypeError):
                            raise InvalidParameterValueError(err_msg)
                    # list and something that can be transformed to a list
                    elif type(default_parameters[k]) == types.ListType: 
                        try:
                            parameters[k] = list(supplied_parameters[k])
                        except TypeError:
                            raise InvalidParameterValueError(err_msg)
                    else:
                        raise InvalidParameterValueError(err_msg)
                else:
                    raise NonExistentParameterError(k, cls)
        return parameters
    
    @classmethod
    def translate(cls, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        parameters = cls.checkParameters(parameters, with_defaults=False)
        native_parameters = {}
        for name in parameters:
            D = cls.translations[name]
            pname = D['translated_name']
            try:
                pval = eval(D['forward_transform'], globals(), parameters)
            except NameError, errmsg:
                raise Exception("Problem translating %s in %s. Transform: %s. Parameters: %s. %s" \
                                % (pname, cls.__name__, D['forward_transform'], parameters, errmsg))
            except ZeroDivisionError:
                pval = 1e30 # this is about the highest value hoc can deal with
            native_parameters[pname] = pval
        return native_parameters
    
    @classmethod
    def reverse_translate(cls, native_parameters):
        """Translate simulator-specific model parameters to standardized parameters."""
        standard_parameters = {}
        for name,D  in cls.translations.items():
            try:
                standard_parameters[name] = eval(D['reverse_transform'], {}, native_parameters)
            except NameError:
                raise Exception("%s in %s. Transform: %s. Parameters: %s." \
                                % (name, cls.__name__, D['reverse_transform'], native_parameters))
        return standard_parameters

    @classmethod
    def simple_parameters(cls):
        """Return a list of parameters for which there is a one-to-one
        correspondance between standard and native parameter values."""
        return [name for name in cls.translations if cls.translations[name]['forward_transform'] == name]

    @classmethod
    def scaled_parameters(cls):
        """Return a list of parameters for which there is a unit change between
        standard and native parameter values."""
        return [name for name in cls.translations if "float" in cls.translations[name]['forward_transform']]
    
    @classmethod
    def computed_parameters(cls):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return [name for name in cls.translations if name not in cls.simple_parameters()+cls.scaled_parameters()]
        
    def update_parameters(self, parameters):
        """
        update self.parameters with those in parameters 
        """
        self.parameters.update(self.translate(parameters))
        
    def describe(self, template='standard'):
        return str(self)
    

class StandardCellType(StandardModelType):
    """Base class for standardized cell model classes."""

    synapse_types = ('excitatory', 'inhibitory')


class IF_curr_alpha(StandardCellType):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    default_parameters = {
        'v_rest'     : -65.0,   # Resting membrane potential in mV. 
        'cm'         :   1.0,   # Capacity of the membrane in nF
        'tau_m'      :  20.0,   # Membrane time constant in ms.
        'tau_refrac' :   0.0,   # Duration of refractory period in ms. 
        'tau_syn_E'  :   0.5,   # Rise time of the excitatory synaptic alpha function in ms.
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
    exponentially-decaying post-synaptic conductance."""
    
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

class IF_cond_exp_gsfa_grr(StandardCellType):
    """Linear leaky integrate and fire model with fixed threshold,
    decaying-exponential post-synaptic conductance, conductance based spike-frequency adaptation,
    and a conductance-based relative refractory mechanism.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond mean-adaptation
    and renewal theories. Neural Computation 19: 2958-3010.

    See also: EIF_cond_alpha_isfa_ista

    """
    
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
        'tau_sfa'    : 100.0,   # Time constant of spike-frequency adaptation in ms
        'e_rev_sfa'  : -75.0,   # spike-frequency adaptation conductance reversal potential in mV
        'q_sfa'      : 15.0,    # Quantal spike-frequency adaptation conductance increase in nS
        'tau_rr'     : 2.0,     # Time constant of the relative refractory mechanism in ms
        'e_rev_rr'   : -75.0,   # relative refractory mechanism conductance reversal potential in mV
        'q_rr'       : 3000.0   # Quantal relative refractory conductance increase in nS
        
    }

    
class IF_facets_hardware1(StandardCellType):
    """Leaky integrate and fire model with conductance-based synapses and fixed 
    threshold as it is resembled by the FACETS Hardware Stage 1. 
    The following parameters can be assumed for a corresponding software
    simulation: cm = 0.2 nF, tau_refrac = 1.0 ms, e_rev_E = 0.0 mV.  
    For further details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    
    default_parameters = {
        'g_leak'    :   40.0,     # nS
        'tau_syn_E' :   30.0,     # ms
        'tau_syn_I' :   30.0,     # ms
        'v_reset'   :  -80.0,     # mV
        'e_rev_I'   :  -80.0,     # mV,
        'v_rest'    :  -65.0,     # mV
        'v_thresh'  :  -55.0      # mV
    }


class HH_cond_exp(StandardCellType):
    """Single-compartment Hodgkin-Huxley model."""
    
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

class EIF_cond_alpha_isfa_ista(StandardCellType):
    """Exponential integrate and fire neuron with spike triggered and
    sub-threshold adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model
    as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr

    """
    
    default_parameters = {
        'v_init'    : -70.6,  # Initial membrane potential in mV
        'w_init'    : 0.0,    # Spike-adaptation current in nA
        'cm'        : 0.281,  # Capacity of the membrane in nF
        'tau_refrac': 0.0,    # Duration of refractory period in ms.
        'v_spike'   : 0.0,    # Spike detection threshold in mV.
        'v_reset'   : -70.6,  # Reset value for V_m after a spike. In mV.
        'v_rest'    : -70.6,  # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m'     : 9.3667, # Membrane time constant in ms (nest:Leak conductance in nS.)
        'i_offset'  : 0.0,    # Offset current in nA
        'a'         : 4.0,    # Subthreshold adaptation conductance in nS.
        'b'         : 0.0805, # Spike-triggered adaptation in nA
        'delta_T'   : 2.0,    # Slope factor in mV
        'tau_w'     : 144.0,  # Adaptation time constant in ms
        'v_thresh'  : -50.4,  # Spike initiation threshold in mV (V_th can also be used for compatibility).
        'e_rev_E'   : 0.0,    # Excitatory reversal potential in mV.
        'tau_syn_E' : 5.0,    # Rise time of excitatory synaptic conductance in ms (alpha function).
        'e_rev_I'   : -80.0,  # Inhibitory reversal potential in mV.
        'tau_syn_I' : 5.0,    # Rise time of the inhibitory synaptic conductance in ms (alpha function).
    }


class SpikeSourcePoisson(StandardCellType):
    """Spike source, generating spikes according to a Poisson process."""

    default_parameters = {
        'rate'     : 1.0,     # Mean spike frequency (Hz)
        'start'    : 0.0,     # Start time (ms)
        'duration' : 1e6      # Duration of spike sequence (ms)
    }  
    synapse_types = ()

class SpikeSourceInhGamma(StandardCellType):
    """Spike source, generating realizations of an inhomogeneous gamma process,
    employing the thinning method.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond
    mean-adaptation and renewal theories. Neural Computation 19: 2958-3010.
    """

    default_parameters = {
        'a'        : numpy.array([1.0]), # time histogram of parameter a of a gamma distribution (dimensionless)
        'b'        : numpy.array([1.0]), # time histogram of parameter b of a gamma distribution (seconds)
        'tbins'    : numpy.array([0]),   # time bins of the time histogram of a,b in units of ms
        'rmax'     : 1.0,                # Rate (Hz) of the Poisson process to be thinned, usually set to max(1/b)
        'start'    : 0.0,                # Start time (ms)
        'duration' : 1e6                 # Duration of spike sequence (ms)
    }  
    synapse_types = ()

class SpikeSourceArray(StandardCellType):
    """Spike source generating spikes at the times given in the spike_times array."""
    
    default_parameters = { 'spike_times' : [] } # list or numpy array containing spike times in milliseconds.
    synapse_types = ()    
           
    def __init__(self, parameters):
        if parameters and 'spike_times' in parameters:
            parameters['spike_times'] = numpy.array(parameters['spike_times'], 'float')
        StandardCellType.__init__(self, parameters)

class ModelNotAvailable(object):
    """Not available for this simulator."""
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The %s model is not available for this simulator." % self.__class__.__name__)

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False,
          **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    dt = timestep
    if min_delay > max_delay:
        raise Exception("min_delay has to be less than or equal to max_delay.")
    invalid_extra_params = ('mindelay', 'maxdelay', 'dt')
    for param in invalid_extra_params:
        if param in extra_params:
            raise Exception("%s is not a valid argument for setup()" % param)

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    pass
    
def run(simtime):
    """Run the simulation for simtime ms."""
    pass

def get_current_time():
    """Return the current time in the simulation."""
    pass

def get_time_step():
    """Return the integration time step being used in the simulation.""" 
    pass

def get_min_delay():
    """Return the minimum allowed synaptic delay."""
    raise Exception("common.get_min_delay() must be overridden by a simulator-specific function")

def get_max_delay():
    """Return the maximum allowed synaptic delay."""
    raise Exception("common.get_max_delay() must be overridden by a simulator-specific function")

def num_processes():
    """When running a parallel simulation with MPI, return the number of processors being used."""
    pass

def rank():
    """Return the MPI rank."""
    pass

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, param_dict=None, n=1):
    """Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    pass

def connect(source, target, weight=None, delay=None, synapse_type=None,
            p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    pass

def set(cells, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value."""
    pass

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    pass

def record_v(source, filename):
    """Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    pass

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(object):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    
    def __init__(self, dims, cellclass, cellparams=None, label=None, parent=None):
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
        
        self.dim = dims
        if isinstance(dims, int): # also allow a single integer, for a 1D population
            self.dim = (self.dim,)
        else:
            assert isinstance(dims, tuple), "`dims` must be an integer or a tuple."
        self.label = label
        self.celltype = cellclass
        self.ndim = len(self.dim)
        self.cellparams = cellparams
        self.size = self.dim[0]
        self.parent = parent
        for i in range(1, self.ndim):
            self.size *= self.dim[i]
        ##self.cell = None # to be defined by child, simulator-specific classes
    
    def __getitem__(self, addr):
        """Return a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        pass
    
    def __iter__(self):
        """Iterator over cell ids."""
        return _abstract_method(self)
        
    def addresses(self):
        """Iterator over cell addresses."""
        return _abstract_method(self)
    
    def ids(self):
        """Iterator over cell ids."""
        return self.__iter__()
    
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        return _abstract_method(self)
    
    def __len__(self):
        """Return the total number of cells in the population."""
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
        assert pos_array.shape == (3, self.size)
        self._positions = pos_array.copy() # take a copy in case pos_array is changed later

    positions = property(_get_positions, _set_positions,
                         """A 3xN array (where N is the number of neurons in the Population)
                         giving the x,y,z coordinates of all the neurons (soma, in the
                         case of non-point models).""")
    
    def index(self, n):
        """Return the nth cell in the population (Indexing starts at 0)."""
        return _abstract_method(self)
    
    def nearest(self, position):
        """Return the neuron closest to the specified position."""
        # doesn't always work correctly if a position is equidistant between two
        # neurons, i.e. 0.5 should be rounded up, but it isn't always.
        pos = numpy.array([position]*self.positions.shape[1]).transpose()
        dist_arr = (self.positions - pos)**2
        distances = dist_arr.sum(axis=0)
        nearest = distances.argmin()
        return self.index(nearest)
            
    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        return _abstract_method(self)

    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        return _abstract_method(self)
    
    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        return _abstract_method(self)
    
    def _call(self, methodname, arguments):
        """
        Call the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        return _abstract_method(self)
    
    def _tcall(self, methodname, objarr):
        """
        `Topographic' call. Call the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init", vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        return _abstract_method(self)

    def randomInit(self, rand_distr):
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        return _abstract_method(self)

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        return _abstract_method(self)

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        return _abstract_method(self)

    def printSpikes(self, filename, gather=True, compatible_output=True):
        """
        Write spike times to file.
        
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """        
        return _abstract_method(self)
    

    def getSpikes(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.
        """
        return _abstract_method(self)

    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        
        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        return _abstract_method(self)
    
    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        return _abstract_method(self)
    
    def describe(self, template='standard'):
        """
        Returns a human readable description of the population
        """
        if template == 'standard':
            #lines = ['==== Population $label ====',
            #         '  Dimensions: $dim',
            #         '  Local cells: $n_cells_local',
            #         '  Cell type: $celltype',
            #         '  ID range: $first_id-$last_id',
            #         '  First cell on this node:',
            #         '    ID: $local_first_id',
            #         '    Parameters: $cell_parameters']
            lines = ['------- Population description -------',
                     'Population called $label is made of $n_cells cells [$n_cells_local being local]']
            if self.parent:
                lines += ['This population is a subpopulation of population $parent_label']
            lines += ["-> Cells are aranged on a ${ndim}D grid of size $dim",
                      "-> Celltype is $celltype",
                      "-> ID range is $first_id-$last_id",
                      "-> Cell Parameters used for cell[0] are: "]
            for name, value in self.index(0).get_parameters().items():
                lines += ["    | %-12s: %s" % (name, value)]
            lines += ["--- End of Population description ----"]
            template = "\n".join(lines)
            
        context = self.__dict__.copy()
        first_id = self._local_ids[0]
        context.update(local_first_id=first_id)
        context.update(cell_parameters=first_id.get_parameters())
        context.update(celltype=self.celltype.__class__.__name__)
        context.update(n_cells=len(self))
        context.update(n_cells_local=len(self._local_ids))
        for k in context.keys():
            if k[0] == '_':
                context.pop(k)
                
        if template == None:
            return context
        else:
            return Template(template).substitute(context)
    
    def getSubPopulation(self, cells):
        """
        Returns a sub population from a population object. The shape of cells will
        determine the dimensions of the sub population. cells should contains cells
        member of the parent population.
        Ex z = pop.getSubPopulation([pop[1],pop[3],pop[5]])
        """
        return _abstract_method(self)
    
# ==============================================================================

class Projection(object):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 method='allToAll', method_parameters=None,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentials
                 
        target - string specifying which synapse on the postsynaptic cell to
                 connect to
                 
        If source and/or target are not given, default values are used.
        
        method - string indicating which algorithm to use in determining
                 connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'.
        
        method_parameters - dict containing parameters needed by the connection
        method, although we should allow this to be a number or string if there
        is only one parameter.
        
        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        
        self.pre    = presynaptic_population  # } these really        
        self.source = source                  # } should be
        self.post   = postsynaptic_population # } read-only
        self.target = target                  # }
        self.label  = label
        self.rng    = rng
        self._method = method
        self.synapse_dynamics = synapse_dynamics
        self.connection = None # access individual connections. To be defined by child, simulator-specific classes
        if label is None:
            if self.pre.label and self.post.label:
                self.label = "%s → %s" % (self.pre.label, self.post.label)
    
        # Deal with synaptic plasticity
        self.short_term_plasticity_mechanism = None
        self.long_term_plasticity_mechanism = None
        if self.synapse_dynamics:
            assert isinstance(self.synapse_dynamics, SynapseDynamics), \
              "The synapse_dynamics argument, if specified, must be a SynapseDynamics object, not a %s" % type(synapse_dynamics)
            if self.synapse_dynamics.fast:
                assert isinstance(self.synapse_dynamics.fast, ShortTermPlasticityMechanism)
                self.short_term_plasticity_mechanism = self.synapse_dynamics.fast.native_name
                self._short_term_plasticity_parameters = self.synapse_dynamics.fast.parameters.copy()
            if self.synapse_dynamics.slow:
                assert isinstance(self.synapse_dynamics.slow, STDPMechanism)
                assert 0 <= self.synapse_dynamics.slow.dendritic_delay_fraction <= 1.0
                td = self.synapse_dynamics.slow.timing_dependence
                wd = self.synapse_dynamics.slow.weight_dependence
                possible_models = td.possible_models.intersection(wd.possible_models)
                if len(possible_models) == 1 :
                    self.long_term_plasticity_mechanism = list(possible_models)[0]
                elif len(possible_models) == 0 :
                    raise Exception("No available plasticity models")
                elif len(possible_models) > 1 :
                    if self.synapse_dynamics.slow.model:
                        # addition of the `model` attribute (see r415) is a pragmatic solution
                        # but not an elegant one, and I don't think it should go into a released API
                        # Since the problem of multiple models only seems to appear for NEST
                        # with homogeneous and non-homogeneous versions, it would be better either
                        # for the code to decide itself which to use (would be complex, as
                        # connection creation would have to be deferred to run()) or to have
                        # a global OPTIMIZED variable (possibly set in setup()) - if this was
                        # set True the homogeneous version would be used and later attempts to
                        # change parameters of the synapse would raise Exceptions.
                        if self.synapse_dynamics.slow.model in list(possible_models):
                            self.long_term_plasticity_mechanism = self.synapse_dynamics.slow.model
                        else:
                            print "The particular model %s does not exists !" %self.synapse_dynamics.slow.model
                    else:
                        print "Several stdp models are available for those plastics connections"
                        for model in possible_models:
                            print "--> %s" %model
                        self.long_term_plasticity_mechanism = list(possible_models)[0]
                        print "By default, %s is used" %self.long_term_plasticity_mechanism
                    #raise Exception("Multiple plasticity models available")
                
                #print "Using %s" % self._plasticity_model
                self._stdp_parameters = td.parameters.copy()
                self._stdp_parameters.update(wd.parameters)
            
    def __len__(self):
        """Return the total number of connections."""
        return self.nconn
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self, parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
    
    def _oneToOne(self, parameters=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        return _abstract_method(self)
    
    def _fixedProbability(self, parameters):
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
    
    def _distanceDependentProbability(self, parameters):
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
    
    def _fixedNumberPre(self, parameters):
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
    
    def _fixedNumberPost(self, parameters):
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
    
    def _fromFile(self, parameters):
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
        
    def _fromList(self, conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        # Need to implement parameter parsing here...
        return _abstract_method(self)
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        return _abstract_method(self)
    
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        return _abstract_method(self)
    
    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        return _abstract_method(self)
    
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        return _abstract_method(self)
    
    def setSynapseDynamics(self, param, value):
        """
        Set parameters of the synapse dynamics linked with the projection
        """
        return _abstract_method(self)
    
    def randomizeSynapseDynamics(self, param, rand_distr):
        """
        Set parameters of the synapse dynamics to values taken from rand_distr
        """
        return _abstract_method(self)
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def getWeights(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with zero or None for non-existent
        connections).
        """
        return _abstract_method(self)
    
    def getDelays(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D delay array (with None or 1e12 for non-existent
        connections).
        """
        return _abstract_method(self)
    
    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        return _abstract_method(self)
    
    def printWeights(self, filename, format='list', gather=True):
        """Print synaptic weights to file."""
        weights = self.getWeights(format=format, gather=gather)
        f = open(filename, 'w', DEFAULT_BUFFER_SIZE)
        if format == 'list':
            f.write("\n".join([str(w) for w in weights]))
        elif format == 'array':
            fmt = "%g "*len(self.post) + "\n"
            for row in weights:
                f.write(fmt % tuple(row))
        f.close()
    
    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        bins = numpy.arange(min, max, float(max-min)/nbins)
        return numpy.histogram(self.getWeights(format='list', gather=True), bins) # returns n, bins
    
    def describe(self, template='standard'):
        """
        Returns a human readable description of the projection
        """
        if template == 'standard':
            lines = ["------- Projection description -------",
                     "Projection $label from $pre_label [$pre_n_cells cells] to $post_label [$post_n_cells cells]",
                     "    | Connector : $_method",
                     "    | Weights : $weights",
                     "    | Delays : $delays",
                     "    | Plasticity : $plasticity",
                     "    | Num. connections : $nconn",
                    ]        
            lines += ["---- End of Projection description -----"]
            template = '\n'.join(lines)
        
        context = self.__dict__.copy()
        context.update({
            'nconn': len(self),
            'pre_label': self.pre.label,
            'post_label': self.post.label,
            'pre_n_cells': self.pre.size,
            'post_n_cells': self.post.size,
            'weights': str(self._method.weights),
            'delays': str(self._method.delays),
        })
        if self.synapse_dynamics:
            context.update(plasticity=self.synapse_dynamics.describe())
        else:
            context.update(plasticity='None')
            
        if template == None:
            return context
        else:
            return Template(template).substitute(context)


# ==============================================================================
#   Connection method classes
# ==============================================================================

class Connector(object):
    """Base class for Connector classes."""
    
    def __init__(self, weights=0.0, delays=None, check_connections=False):
        self.w_index = 0 # should probably use a generator
        self.d_index = 0 # rather than storing these values
        self.weights = weights
        self.delays = delays
        self.check_connections = check_connections
        if delays is None:
            self.delays = get_min_delay()
    
    def connect(self, projection):
        """Connect all neurons in ``projection``"""
        _abstract_method(self)
        
    def getWeights(self, N):
        """
        Returns the next N weight values
        """
        if isinstance(self.weights, random.RandomDistribution):
            weights = numpy.array(self.weights.next(N))
        elif isinstance(self.weights, (int, float)):
            weights = numpy.ones((N,))*float(self.weights)
        elif hasattr(self.weights, "__len__"):
            weights = self.weights[self.w_index:self.w_index+N]
        else:
            raise Exception("weights is of type %s" % type(self.weights))
        self.w_index += N
        return weights
    
    def getDelays(self, N, start=0):
        """
        Returns the next N delay values
        """
        if isinstance(self.delays, random.RandomDistribution):
            delays = numpy.array(self.delays.next(N))
        elif isinstance(self.delays, (int, float)):
            delays = numpy.ones((N,))*float(self.delays)
        elif hasattr(self.delays, "__len__"):
            delays = self.delays[self.d_index:self.d_index+N]
        else:
            raise Exception("delays is of type %s" % type(self.delays))
        if self.check_connections:
            assert numpy.all(delays >= get_min_delay()), \
            "Delay values must be greater than or equal to the minimum delay %g. The smallest delay is %g." % (get_min_delay(), delays.min())
            assert numpy.all(delays <= get_max_delay()), \
            "Delay values must be less than or equal to the maximum delay %s. The largest delay is %s" % (get_max_delay(), delays.max())                                                                                              
        self.d_index += N
        return delays
    
    
class AllToAllConnector(Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    
    def __init__(self, allow_self_connections=True, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections


class FromListConnector(Connector):
    """
    Make connections according to a list.
    """
    
    def __init__(self, conn_list):
        Connector.__init__(self, 0., 0.)
        self.conn_list = conn_list        


class FromFileConnector(Connector):
    """
    Make connections according to a list read from a file.
    """
    
    def __init__(self, filename, distributed=False):
        Connector.__init__(self, 0., 0.)
        self.filename = filename
        self.distributed = distributed
       
        
class FixedNumberPostConnector(Connector):
    """
    Each pre-synaptic neuron is connected to exactly n post-synaptic neurons
    chosen at random.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None):
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
    Each post-synaptic neuron is connected to exactly n pre-synaptic neurons
    chosen at random.
    """
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
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
    
    def __init__(self, p_connect, allow_self_connections=True, weights=0.0, delays=None):
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
    
    def __init__(self, d_expression, axes=None, scale_factor=1.0, offset=0.0,
                 periodic_boundaries=False, allow_self_connections=True,
                 weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        assert isinstance(d_expression, str)
        try:
            d = 0; assert 0 <= eval(d_expression), eval(d_expression)
            d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError:
            raise Exception("Error in the distance expression %s" %d_expression)
        self.d_expression = d_expression
        # We will use the numpy functions, so we need to parse the function
        # given by the user to look for some key function and add numpy
        # in front of them (or add from numpy import *)
        #func = ['exp','log','sin','cos','cosh','sinh','tan','tanh']
        #for item in func:
            #self.d_expression = self.d_expression.replace(item,"numpy.%s" %item)
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
    For specifying synapse short-term (faciliation, depression) and long-term
    (STDP) plasticity. To be passed as the `synapse_dynamics` argument to
    `Projection.__init__()` or `connect()`.
    """
    
    def __init__(self, fast=None, slow=None):
        self.fast = fast
        self.slow = slow
    
    def describe(self, template='standard'):
        if template == 'standard':
            lines = ["Short-term plasticity mechanism: $slow",
                     "Long-term plasticity mechanism: $fast"]
            template = "\n".join(lines)
        context = {'fast': self.fast.describe(),
                   'slow': self.slow.describe()}
        if template == None:
            return context
        else:
            return Template(template).substitute(context)
        
        
class ShortTermPlasticityMechanism(StandardModelType):
    """Abstract base class for models of short-term synaptic dynamics."""
    
    def __init__(self):
        _abstract_method(self)


class STDPMechanism(object):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0, model=None):
        self.timing_dependence = timing_dependence
        self.weight_dependence = weight_dependence
        self.voltage_dependence = voltage_dependence
        self.dendritic_delay_fraction = dendritic_delay_fraction
        self.model = model # see comment in Projection.__init__()


class TsodyksMarkramMechanism(ShortTermPlasticityMechanism):
    """ """
    default_parameters = {
        'U': 0.5,   # use parameter
        'tau_rec': 100.0, # depression time constant (ms)
        'tau_facil': 0.0,   # facilitation time constant (ms)
        'u0': 0.0,  # }
        'x0': 1.0,  # } initial values
        'y0': 0.0   # }
    }
    
    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        pass

        
class STDPWeightDependence(StandardModelType):
    """Abstract base class for models of STDP weight dependence."""
    
    def __init__(self):
        _abstract_method(self)


class STDPTimingDependence(StandardModelType):
    """Abstract base class for models of STDP timing dependence (triplets, etc)"""
    
    def __init__(self):
        _abstract_method(self)
        

class AdditiveWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.
    """
    default_parameters = {
        'w_min':   0.0,
        'w_max':   1.0,
        'A_plus':  0.01,
        'A_minus': 0.01
    }
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        pass #_abstract_method(self)


class MultiplicativeWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    default_parameters = {
        'w_min'  : 0.0,
        'w_max'  : 1.0,
        'A_plus' : 0.01,
        'A_minus': 0.01,
    }
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        pass
    

class AdditivePotentiationMultiplicativeDepression(STDPWeightDependence):
    """
    For depression, Dw propto w-w_min
    For potentiation, Dw constant
    (van Rossum rule?)
    """

    default_parameters = {
        'w_min'  : 0.0,
        'w_max'  : 1.0,
        'A_plus' : 0.01,
        'A_minus': 0.01,
    }

    def __init__(self, w_min=0.0,  w_max=1.0, A_plus=0.01, A_minus=0.01):
        pass

    
class GutigWeightDependence(STDPWeightDependence):
    
    default_parameters = {
        'w_min'   : 0.0,
        'w_max'   : 1.0,
        'A_plus'  : 0.01,
        'A_minus' : 0.01,
        'mu_plus' : 0.5,
        'mu_minus': 0.5
    }

    def __init__(self, w_min=0.0,  w_max=1.0, A_plus=0.01, A_minus=0.01,mu_plus=0.5,mu_minus=0.5):
        pass

# Not yet implemented for any module
#class PfisterSpikeTripletRule(STDPTimingDependence):
#    pass


class SpikePairRule(STDPTimingDependence):
    
    default_parameters = {
        'tau_plus':  20.0,
        'tau_minus': 20.0,
    }
    
    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        pass #_abstract_method(self)


# ==============================================================================
#   Utility classes
# ==============================================================================
   
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

# ==============================================================================
