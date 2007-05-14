"""
Defines the PyNN classes and functions, and hence the FACETS API.
The simulator-specific classes should inherit from these and have the same
arguments.
$Id$
"""
__version__ = "$Revision$"

import types, time, copy

class InvalidParameterValueError(Exception): pass
class NonExistentParameterError(Exception): pass
class InvalidDimensionsError(Exception): pass
class ConnectionError(Exception): pass

dt = 0.1


# ==============================================================================
#   Utility classes
# ==============================================================================

class ID(int):
    """
    This class is experimental. The idea is that instead of storing ids as
    integers, we store them as ID objects, which allows a syntax like:
      p[3,4].set('tau_m',20.0)
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def __init__(self,n):
        int.__init__(n)
        self._position  = None
        self._cellclass = None
        self._hocname   = None
        # The cellclass can be a global attribute of the ID object, but
        # it may be discussed: 
        # The problem is that a call to the low-level funcitons set() and get() will need
        # the cellclass to work. So we have to choose if we want to store that information in the ID
        # object (as an attribute for example) or if we want to type it each time we need a call to set()
        # or get() : p[2,3].set(SpikeSourceArray, {'spike_train' : {}}).

    def set(self,param,val=None):
        pass
    
    def get(self,param):
        pass

    def setCellClass(self, cellclass):
        self._cellclass = cellclass    
    
    # Here is a proposal to manage the physical position of the cell, as an
    # attribute of the ID class. Those positions can be used by functions such
    # as _distantDependantProbability(), setTopographicDelay()...
    def setPosition(self,pos):
        self._position = pos
        
    def getPosition(self):
        return self._position



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
        'tau_syn'    :   5.0,   # Rise time of the synaptic alpha function in ms.
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
        'tau_syn_E'  : 5.0,     # Rise time of the excitatory synaptic alpha function in ms.
        'tau_syn_I'  : 5.0,     # Rise time of the inhibitory synaptic alpha function in ms.
        'e_rev_E'    : 0.0,     # Reversal potential for excitatory input in mV
        'e_rev_I'    : -70.0,   # Reversal potential for inhibitory input in mV
        'v_thresh'   : -50.0,   # Spike threshold in mV.
	'v_reset'    : -65.0,   # Reset potential after a spike in mV.
	'i_offset'   : 0.0,     # Offset current in nA
        'v_init'     : -65.0,   # Membrane potential in mV at t = 0
    }

class SpikeSourcePoisson(StandardCellType):
    """Spike source, generating spikes according to a Poisson process."""

    default_parameters = {
        'rate'     : 0.0,       # Mean spike frequency (Hz)
        'start'    : 0.0,       # Start time (ms)
        'duration' : 1e9        # Duration of spike sequence (ms)
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
    Weights should be in nA or uS."""
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
            print "Converting integer dims to tuple"
            self.dim = (self.dim,)
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
    
    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size
    
    def set(self,param,val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        pass

    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        pass
    
    def rset(self,parametername,rand_distr):
        """
        'Random' set. Sets the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        pass
    
    def _call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        pass
    
    def _tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        pass

    def randomInit(self,rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        pass

    def record(self,record_from=None,rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids (e.g., (i,j,k) tuple for a 3D population)
        of the cells to record.
        """
        pass

    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        pass

    def printSpikes(self,filename,gather=True, compatible_output=True):
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
        pass
    
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
        pass
    
    def meanSpikeCount(self,gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        pass

# ==============================================================================

class Projection:
    """
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 method='allToAll', methodParameters=None,
                 source=None, target=None, label=None, rng=None):
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
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within methodParameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        
        self.pre    = presynaptic_population  # } these really        
        self.source = source                  # } should be
        self.post   = postsynaptic_population # } read-only
        self.target = target                  # }
        if label:
            self.label = label
        self.rng = rng
        self.connection = None # access individual connections. To be defined by child, simulator-specific classes
    
    def __len__(self):
        """Return the total number of connections."""
        return self.nconn
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None,synapse_type=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
    
    def _oneToOne(self,synapse_type=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        pass
    
    def _fixedProbability(self,parameters,synapse_type=None):
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
    
    def _distanceDependentProbability(self,parameters,synapse_type=None):
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
    
    def _fixedNumberPre(self,parameters,synapse_type=None):
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
    
    def _fixedNumberPost(self,parameters,synapse_type=None):
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
    
    def _fromFile(self,parameters,synapse_type=None):
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
        
    def _fromList(self,conn_list,synapse_type=None):
        """
        Read connections from a list of tuples,
        containing ['src[x,y]', 'tgt[x,y]', 'weight', 'delay']
        """
        # Need to implement parameter parsing here...
        pass
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        pass
    
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        pass
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        pass
    
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        pass
    
    def setThreshold(self,threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        # This is a bit tricky, because in NEST the spike threshold is a
        # property of the cell model, whereas in NEURON it is a property of the
        # connection (NetCon).
        pass
    
    
    # --- Methods relating to synaptic plasticity ------------------------------
    
    def setupSTDP(self,stdp_model,parameterDict):
        """Set-up STDP."""
        pass
    
    def toggleSTDP(self,onoff):
        """Turn plasticity on or off."""
        pass
    
    def setMaxWeight(self,wmax):
        """Note that not all STDP models have maximum or minimum weights."""
        pass
    
    def setMinWeight(self,wmin):
        """Note that not all STDP models have maximum or minimum weights."""
        pass
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self,filename,gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        pass
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        pass
    
    def weightHistogram(self,min=None,max=None,nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        pass


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
