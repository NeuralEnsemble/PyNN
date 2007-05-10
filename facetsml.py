"""
FACETS-ML implementation of the PyNN API.
$Id$
"""

import common
import numpy, types, sys, shutil
import RandomArray
from xml.dom import *
from xml.dom.minidom import *
from xml.dom.ext import *

        
open_files = []

dt = 0.1
xmldoc = Document()

"""
warning :
	in order to write xml in a format which respects the namespaces, you must use xml.dom.ext.PrettyPrint
namespaces allowed are :
		neuromlNode.setAttribute('xmlns:net','http://morphml.org/networkml/schema')
		neuromlNode.setAttribute('xmlns:mml','http://morphml.org/morphml/schema')
		neuromlNode.setAttribute('xmlns:meta','http://morphml.org/metadata/schema')
		neuromlNode.setAttribute('xmlns:bio','http://morphml.org/biophysics/schema')
		neuromlNode.setAttribute('xmlns:cml','http://morphml.org/channelml/schema')

"""

def initDocument(parentElementNS,parentElementName,prefix=''):
	"""
	create the root element <neuroml> if doesn't exist
	and the specified parentElement just below <neuroml> if doesn't exist
	returns the parentElement node
	"""
	neuromlNodes = xmldoc.getElementsByTagNameNS('http://morphml.org/neuroml/schema','neuroml')
	#if the <neuroml> markup is not yet created
	if(neuromlNodes.length == 0):
		#seems createElementNS doesn't create the xmlns attribute
		neuromlNode = xmldoc.createElementNS('http://morphml.org/neuroml/schema','neuroml')
		xmldoc.appendChild(neuromlNode)
	else:
		neuromlNode = neuromlNodes[0]
	parentElementNodes = neuromlNode.getElementsByTagNameNS(parentElementNS,parentElementName)
	if(parentElementNodes.length == 0):
		if(prefix == ''):
			parentElementNode = xmldoc.createElementNS(parentElementNS,parentElementName)
		else:
			parentElementNode = xmldoc.createElementNS(parentElementNS,prefix + ":" + parentElementName)
		neuromlNode.appendChild(parentElementNode)
	else:
		parentElementNode = parentElementNodes[0]
	return parentElementNode


# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

class StandardCells: # do we need this? The FACETS-ML names and the PyNN names should be the same
    """
    This is a static class which which contains one method for each of the
    standard FACETS cell models. Each method:
      (i) has a .nest_name attribute which is the NEST-specific name for the model
      (ii) takes a dictionary whose keys are the standard parameter names
      (iii) returns a dictionary whose keys are the NEST-specific parameter
         names. This dictionary also contains any extra, NEST-only parameters.
    """
    
    def IF_curr_alpha(parameterDict):
        """
        Integrate-and-fire cell with alpha-shaped post-synaptic current.
        
        The keys in parameterDict must come from this list:
          'vrest', 'cm', 'tau_m', 'tau_refrac', 'tau_syn', 'v_thresh'
        Units: v* (mV), cm (nF), tau* (ms)
        Any required parameters not in parameterDict will be given default values.
        
        Returns a parameter dictionary with NEST-specific parameter names.
        """
        global dt
        
        parameters = common.default_values['IF_curr_alpha']
        if parameterDict:
            for k in parameters.keys():
                if parameterDict.has_key(k):
                    parameters[k] = parameterDict[k]
        if parameters['v_reset'] != parameters['v_rest']:
            raise "It is not possible to make v_reset different from v_rest in iaf_neuron."
        translated_parameters = {
            'U0'         : parameters['v_rest'],
            'C'          : parameters['cm']*1000.0, # C is in pF, cm in nF
            'Tau'        : parameters['tau_m'],
            'TauR'       : max(dt,parameters['tau_refrac']),
            'TauSyn'     : parameters['tau_syn'],
            'Theta'      : parameters['v_thresh'],
            'I0'         : parameters['i_offset']*1000.0, # I0 is in pA, i_offset in nA
            'LowerBound' : -1000.0 }
        return translated_parameters
     
    def SpikeSourcePoisson(parameterDict):
        """
        Spike source, generating spikes according to a Poisson process.
        
        The keys in parameterDict must come from this list:
          'rate', 'start', 'duration'
        Units: rate (Hz), start (ms), duration (ms)
        Any required parameters not in parameterDict will be given default values.
        
        Returns a parameter dictionary with NEST-specific parameter names
        """
        
        parameters = common.default_values['SpikeSourcePoisson']
        if parameterDict:
            for k in parameters.keys():
                if parameterDict.has_key(k):
                    parameters[k] = parameterDict[k]
        translated_parameters = {
            'rate'     : parameters['rate'],
            'start'    : parameters['start'],
            'duration' : parameters['duration'],
            'origin'   : 1.0 }
        return translated_parameters   
    
    def SpikeSourceArray(parameterDict):
        """
        Spike source generating spikes at the times given in the spike_times array.
        
        parameterDict must contain this key: spike_times, whose value should be
        a list or numpy array containing spike times in milliseconds.

        Returns a parameter dictionary with NEST-specific parameter names
        """
        parameters = common.default_values['SpikeSourceArray']
        for k in parameters.keys():
            if parameterDict.has_key(k):
                parameters[k] = parameterDict[k]
        translated_parameters = {
            'spike_times' : parameters['spike_times']
        }
        return translated_parameters
            
    
    setattr(IF_curr_alpha,'nest_name','iaf_neuron')
    setattr(SpikeSourcePoisson,'nest_name','poisson_generator')
    setattr(SpikeSourceArray,'nest_name','spike_generator')
    IF_curr_alpha = staticmethod(IF_curr_alpha)
    SpikeSourcePoisson = staticmethod(SpikeSourcePoisson)
    SpikeSourceArray = staticmethod(SpikeSourceArray)

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1):
    """Should be called at the very beginning of a script."""
    global dt
    dt = timestep
    raise "Not yet implemented"

def end():
    """Do any necessary cleaning up before exiting."""
    raise "Not yet implemented"

def run(simtime):
    """Run the simulation for simtime ms."""
    raise "Not yet implemented"

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    raise "Not yet implemented"
    

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(celltype,paramDict=None,n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    assert n > 0, 'n must be a positive integer'
    translate = getattr(StandardCells,celltype)    
    cell_ids = None
    raise "Not yet implemented"
    if n == 1:
        return cell_ids[0]
    else:
        return cell_ids

def connect(source,target,weight=None,delay=None,p=1):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p."""
    global dt
    if weight is None:
        weight = 0.0
    if delay is None:
        delay = dt
    if type(source) != types.ListType and type(target) != types.ListType:
        connect_id = None
    else:
        connect_id = []
        if type(source) != types.ListType:
            source = [source]
        if type(target) != types.ListType:
            target = [target]
        for src in source:
            src = pynest.getAddress(src)
            for tgt in target:
                tgt = pynest.getAddress(tgt)
                if int(p) == 1 or RandomArray.uniform(0,1)<p:
                    connect_id += [None]
    raise "Not yet implemented"
    return connect_id

def set(cells,celltype,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    celltype must be supplied for doing translation of parameter names."""
    translate = getattr(StandardCells,celltype) 
    if val:
        param = {param:val}
    if type(cells) != types.ListType:
        cells = [cells]
    raise "Not yet implemented"

def record(src,filename):
    """Record spikes to a file. src can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    raise "Function not yet implemented."

def record_v(source,filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    if type(source) == types.ListType:
        source = [pynest.getAddress(src) for src in source]
    else:
        source = [pynest.getAddress(source)]
    for src in source:
        None
    raise "Function not yet implemented."
    

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    nPop = 0
    
    def __init__(self,dims,cellclass,cellparams=None,label=None):
        """
        dims should be a tuple containing the population dimensions, or a single
          integer, for a one-dimensional population.
          e.g., (10,10) will create a two-dimensional population of size 10x10.
        celltype should be a string - the name of the neuron model class that
          makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        label is an optional name for the population.
	
	example of NeuroML (completeNetwork.xml with CellGroupC example added) :
	<net:populations>
		<net:population name="CellGroupA">
			<net:cell_type>CellA</net:cell_type>
			<net:instances>
				<net:instance id="0"><net:location x="0" y="0" z="0"/></net:instance>
				<net:instance id="1"><net:location x="0" y="10" z="0"/></net:instance>
				<net:instance id="2"><net:location x="0" y="20" z="0"/></net:instance>
			</net:instances>
		</net:population>
		<net:population name="CellGroupB">
			<net:cell_type>CellA</net:cell_type>
			<net:instances>
				<net:instance id="0"><net:location x="0" y="100" z="0"/></net:instance>
				<net:instance id="1"><net:location x="20" y="100" z="0"/></net:instance>
			</net:instances>
		</net:population>
		<net:population name="CellGroupC">
			<net:cell_type>CellC</net:cell_type>
			<net:pop_location reference="aeag">
				<net:grid_arrangement>
					<net:rectangular_location name="aefku">
						<meta:corner x="0" y="0" z="0"/>
						<meta:size depth="10" height="100" width="100"/>
					</net:rectangular_location>
					<net:spacing x="10" y="10" z="10"/>
				</net:grid_arrangement>
			</net:pop_location>
		</net:population>
	</net:populations>
	
	
	
        """
        
        common.Population.__init__(self,dims,cellclass,cellparams,label)
                     
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
	
	
	populationsNode = initDocument('http://morphml.org/networkml/schema','populations','net')
	
	populationNode = xmldoc.createElementNS('http://morphml.org/networkml/schema','net:population')
	populationNode.setAttribute('name',label)
	populationsNode.appendChild(populationNode)
	
	cell_typeNode = xmldoc.createElementNS('http://morphml.org/networkml/schema','net:cell_type')
	#coming from neuron.py
	if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cellparams = self.celltype.parameters
            hoc_name = self.celltype.hoc_name
        elif isinstance(cellclass, str): # not a standard model
            hoc_name = cellclass
        #end of coming
	
	cell_typeTextNode = xmldoc.createTextNode(hoc_name)
	cell_typeNode.appendChild(cell_typeTextNode)
	populationNode.appendChild(cell_typeNode)
	"""
	the minimal neuroml to add there is :
	   <net:pop_location reference="aReference">
                <net:grid_arrangement>
                    <net:rectangular_location name="aName">
                        <meta:corner x="0" y="0" z="0"/>
                        <meta:size depth="10" height="100" width="100"/>
                    </net:rectangular_location>
                    <net:spacing x="10" y="10" z="10"/>
                </net:grid_arrangement>
                
            </net:pop_location>
	"""
	pop_locationNode = xmldoc.createElementNS('http://morphml.org/networkml/schema','net:pop_location')
	pop_locationNode.setAttribute('reference','aReference')
	populationNode.appendChild(pop_locationNode)
	
	grid_arrangementNode = xmldoc.createElementNS('http://morphml.org/networkml/schema','net:grid_arrangement')
	pop_locationNode.appendChild(grid_arrangementNode)
	
	rectangular_locationNode = xmldoc.createElementNS('http://morphml.org/networkml/schema','net:rectangular_location')
	rectangular_locationNode.setAttribute('name','aName')
	grid_arrangementNode.appendChild(rectangular_locationNode)
	
	cornerNode = xmldoc.createElementNS('http://morphml.org/metadata/schema','meta:corner')
	cornerNode.setAttribute('x','0')
	cornerNode.setAttribute('y','0')
	cornerNode.setAttribute('z','0')
	rectangular_locationNode.appendChild(cornerNode)
	
	sizeNode = xmldoc.createElementNS('http://morphml.org/metadata/schema','meta:size')
	#neuroml is always in 3D adding 0 for non covered dimensions
	sizeNode.setAttribute('depth',str(10*dims[0]))
	sizeNode.setAttribute('height',str(10*dims[1]))
	if(dims.__len__() > 2):
		sizeNode.setAttribute('width',str(10*dims[2]))
	else:
		sizeNode.setAttribute('width','0')
	rectangular_locationNode.appendChild(sizeNode)
	
	spacingNode = xmldoc.createElementNS('http://morphml.org/networkml/schema','net:spacing')
	spacingNode.setAttribute('x','10')
	spacingNode.setAttribute('y','10')
	spacingNode.setAttribute('z','10')
	grid_arrangementNode.appendChild(spacingNode)
	
	
	#cellparams would be defined in a <cell> markup which would define precisely the neuron model
	
	
        #raise "Not yet implemented."
        
        
        Population.nPop += 1
	PrettyPrint(xmldoc)
	
        
    def set(self,param,val):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        e.g. p.set("tau",20.0).
             p.set({'tau':20,'v_rest':-65})
        """
        raise "Method not yet implemented."
        
    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        raise "Method not yet implemented"
    
    def rset(self,parametername,randomobj):
        """
        'Random' set. Sets the value of parametername to a value taken from
        the randomobj Random object.
        """
        raise "Method not yet implemented"
    
    def call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        raise "Method not yet implemented"
    
    def tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        raise "Method not yet implemented"

    def record(self,record_from=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random - or a list containing the ids (e.g., (i,j,k) tuple for a 3D
        population) of the cells to record.
        """
        raise "Method not yet implemented"

    def record_v(self,record_from=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random - or a list containing the ids (e.g., (i,j,k) tuple for a 3D
        population) of the cells to record.
        """
        raise "Method not yet implemented"
    
    
    def printSpikes(self,filename):
        """
        Prints spike times to file in the two-column format
        "spiketime cell_id" where cell_id is the index of the cell counting
        along rows and down columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        """
        raise "Method not yet implemented"

    def meanSpikeCount(self):
        """
        Returns the mean number of spikes per neuron.
        """
        raise "Method not yet implemented"

    def randomInit(self,randobj):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        raise "Method not yet implemented"
    
    def print_v(self,filename):
        """
        Write membrane potential traces to file. Assumes that the cell class
        defines an array vrecord that is used to record membrane potential.
        """
        raise "Method not yet implemented"
        
    
class Projection(common.Projection):
    """
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    def __init__(self,presynaptic_population,postsynaptic_population,method='allToAll',methodParameters=None,source=None,target=None,label=None,rng=None):
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
        it is probably more convenient to specify a Random object here rather
        than within methodParameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,methodParameters,source,target,label,rng)
        self.connection = []
        self._targets = []
        self._sources = []
        connection_method = getattr(self,'_%s' % method)
        self.nconn = connection_method(methodParameters)
        
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        raise "Method not yet implemented"
        return len(presynaptic_neurons) * len(postsynaptic_neurons)
    
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
        raise "Method not yet implemented"
    
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
         
        raise "Method not yet implemented"
    
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
        raise "Method not yet implemented"
    
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
        raise "Method not yet implemented"
    
    def _fixedNumberPost(self,parameters): #CHEAT CHEAT CHEAT
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
        
        raise "Method not yet implemented"
    
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
        raise "Method not yet implemented"
        
    def _fromList(self,parameters):
        """
        Read connections from a list of lists, or somesuch...
        """
        # Need to implement parameter parsing here...
        raise "Method not yet implemented"
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        if type(w) == types.FloatType or type(w) == types.IntType:
            w = w*1000
        for src,tgt in zip(self._sources,self._targets):
            # set weight
            raise "Method not yet implemented"
        else:
            raise "Method needs changing to reflect the new API" # (w can be an array)
    
    def randomizeWeights(self,rng):
        """
        Set weights to random values taken from rng.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        raise "Method not yet implemented"
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        if type(d) == types.FloatType or type(d) == types.IntType:
            for src,tgt in zip(self._sources,self._targets):
                # set delays
                raise "Method not yet implemented"
        else:
            raise "Method needs changing to reflect the new API" # (d can be an array)
    
    def randomizeDelays(self,rng):
        """
        Set delays to random values taken from rng.
        """
        raise "Method not yet implemented"
    
    def setThreshold(self,threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        raise "Method not yet implemented"
    
    
    # --- Methods relating to synaptic plasticity ------------------------------
    
    def setupSTDP(self,stdp_model,parameterDict):
        """Set-up STDP."""
        raise "Method not yet implemented"
    
    def toggleSTDP(self,onoff):
        """Turn plasticity on or off."""
        raise "Method not yet implemented"
    
    def setMaxWeight(self,wmax):
        """Note that not all STDP models have maximum or minimum weights."""
        raise "Method not yet implemented"
    
    def setMinWeight(self,wmin):
        """Note that not all STDP models have maximum or minimum weights."""
        raise "Method not yet implemented"
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self,filename):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        raise "Method not yet implemented"
    
    def printWeights(self,filename,format=None):
        """Print synaptic weights to file."""
        raise "Method not yet implemented"
    
    def weightHistogram(self,min=None,max=None,nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        raise "Method not yet implemented"
 
# ==============================================================================
#   Utility classes
# ==============================================================================
   
Timer = common.Timer  # not really relevant here except for timing how long it takes
                      # to write the XML file. Needed for API consistency.

# ==============================================================================

class Random:
    """Wrapper class for random number generators. The idea is to be able to use
    either simulator-native rngs, which may be more efficient, or a standard
    python rng, e.g. numpy.Random, which would allow the same random numbers to
    be used across different simulators, or simply to read externally-generated
    numbers from files."""
    
    nRand = 0
    
    def __init__(self,type='default',distribution='uniform',label=None,seed=123456789):
        """ """
        raise "Not yet implemented."
        
    def next(self,n):
        """Return n random numbers from the distribution."""
        raise "Not yet implemented."