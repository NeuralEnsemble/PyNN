# encoding: utf-8
"""
PyNN-->NeuroML
$Id:$
"""

from pyNN import common
import math
#import numpy, types, sys, shutil
import xml.dom.minidom
import xml.dom.ext

neuroml_url = 'http://morphml.org'
namespace = {'xsi': "http://www.w3.org/2001/XMLSchema-instance",
             'mml':  neuroml_url+"/morphml/schema",
             'net':  neuroml_url+"/networkml/schema",
             'meta': neuroml_url+"/metadata/schema",
             'bio':  neuroml_url+"/biophysics/schema",  
             'cml':  neuroml_url+"/channelml/schema", }

# ==============================================================================
#   Utility classes
# ==============================================================================

class ID(common.ID):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def __init__(self,n):
        common.ID.__init__(self,n)

# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

def build_node(name_, text=None, **attributes):
    # we call the node name 'name_' because 'name' is a common attribute name (confused? I am)
    ns, name_ = name_.split(':')
    if ns:
        node = xmldoc.createElementNS(namespace[ns], "%s:%s" % (ns,name_))
    else:
        node = xmldoc.createElement(name_)
    for attr,value in attributes.items():
        node.setAttribute(attr, str(value))
    if text:
        node.appendChild(xmldoc.createTextNode(text))
    return node

def build_parameter_node(name, value):
        param_node = build_node('bio:parameter', value=value)
        if name:
            param_node.setAttribute('name', name)
        group_node = build_node('bio:group', 'all')
        param_node.appendChild(group_node)
        return param_node

class IF_base(object):
    """Base class for integrate-and-fire neuron models."""
    
    def define_morphology(self):
        segments_node = build_node('mml:segments')
        soma_node = build_node('mml:segment',id=0, name="Soma", cable=0)
        # L = 100  diam = 1000/PI: gives area = 10³ cm²
        soma_node.appendChild(build_node('mml:proximal', x=0, y=0, z=0, diameter=1000/math.pi))
        soma_node.appendChild(build_node('mml:proximal', x=0, y=0, z=100, diameter=1000/math.pi))
        segments_node.appendChild(soma_node)
        
        cables_node   = build_node('mml:cables')
        soma_node = build_node('mml:cable', id=0, name="Soma")
        soma_node.appendChild(build_node('meta:group','all'))
        cables_node.appendChild(soma_node)
        return segments_node, cables_node
        
    def define_biophysics(self):
        # L = 100  diam = 1000/PI  // 
        biophys_node  = build_node(':biophysics', units="Physiological Units")
        ifnode        = build_node('bio:mechanism', name='IandF', type='Channel Mechanism')
        passive_node  = build_node('bio:mechanism', name='pas', type='Channel Mechanism')
        # g_max = 10⁻³cm/tau_m  // cm(nF)/tau_m(ms) = G(µS) = 10⁻⁶G(S). Divide by area (10³) to get factor of 10⁻³
        gmax = str(1e-3*self.parameters['cm']/self.parameters['tau_m'])
        passive_node.appendChild(build_parameter_node('gmax', gmax))
        cm_node       = build_node('bio:specificCapacitance')
        cm_node.appendChild(build_parameter_node('', str(self.parameters['cm'])))  # units?
        Ra_node       = build_node('bio:specificAxialResistance')
        Ra_node.appendChild(build_parameter_node('', "0.1")) # value doesn't matter for a single compartment
        esyn_node     = build_node('bio:mechanism', name="ExcitatorySynapse", type="Channel Mechanism")
        isyn_node     = build_node('bio:mechanism', name="InhibitorySynapse", type="Channel Mechanism")
        
        for node in ifnode,passive_node, cm_node, Ra_node, esyn_node, isyn_node:
            biophys_node.appendChild(node)
        return biophys_node
        
    def define_channel_types(self):
        ion_node     = build_node('cml:ion', name="non_specific",
                                  charge=1, default_erev=self.parameters['v_rest'])
        
        passive_node = build_node('cml:channel_type', name="pas", density="yes")
        passive_node.appendChild( build_node('meta:notes', "Simple example of a leak/passive conductance") )
        cvr_node = build_node('cml:current_voltage_relation')
        ohmic_node = build_node('cml:ohmic', ion="non_specific")
        gmax = str(1e-3*self.parameters['cm']/self.parameters['tau_m'])
        ohmic_node.appendChild( build_node('cml:conductance', default_gmax=gmax) )
        cvr_node.appendChild(ohmic_node)
        passive_node.appendChild(cvr_node)
        
        ifnode = build_node('cml:channel_type', name="IandF")
        ifnode.appendChild( build_node('meta:notes', "Spike and reset mechanism") )
        cvr_node = build_node('cml:current_voltage_relation')
        ifmech_node = build_node('cml:integrate_and_fire',
                                 threshold=self.parameters['v_thresh'],
                                 t_refrac=self.parameters['tau_refrac'],
                                 v_reset=self.parameters['v_reset'],
                                 g_refrac=0.1) # this value just needs to be 'large enough'
        cvr_node.appendChild(ifmech_node)
        ifnode.appendChild(cvr_node)
        
        return [ion_node, passive_node, ifnode]
            
    def define_synapse_types(self, synapse_type):
        esyn_node = build_node('cml:synapse_type', name="ExcitatorySynapse")
        esyn_node.appendChild( build_node('cml:%s' % synapse_type,
                                          max_conductance="1.0e-5",
                                          rise_time="0",
                                          decay_time=self.parameters['tau_syn_E'],
                                          reversal_potential=self.parameters['e_rev_E'] ) )
        isyn_node = build_node('cml:synapse_type', name="InhibitorySynapse")
        isyn_node.appendChild( build_node('cml:%s' % synapse_type,
                                          max_conductance="1.0e-5",
                                          rise_time="0",
                                          decay_time=self.parameters['tau_syn_I'],
                                          reversal_potential=self.parameters['e_rev_I'] ) )
        return [esyn_node, isyn_node]

    def build_nodes(self):
        cell_node = build_node(':cell', name=self.label)
        doc_node = build_node('meta:notes', "Instance of PyNN %s cell type" % self.__class__.__name__)
        segments_node, cables_node = self.define_morphology()
        biophys_node = self.define_biophysics()
        for node in doc_node, segments_node, cables_node, biophys_node:
            cell_node.appendChild(node)
        
        channel_nodes = self.define_channel_types()
        synapse_nodes = self.define_synapse_types(self.synapse_type)
        channel_nodes.extend(synapse_nodes)

        return cell_node, channel_nodes

# ==============================================================================
#   Standard cells
# ==============================================================================

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses"""
    
    def __init__(self):
        raise Exception('Cell type %s is not available in NeuroML' % self.__class__.__name__)

class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    def __init__(self):
        raise Exception('Cell type %s is not available in NeuroML' % self.__class__.__name__)

class IF_cond_exp(common.IF_cond_exp, IF_base):
    """Leaky integrate and fire model with fixed threshold and 
    decaying-exponential post-synaptic conductance."""
    
    n = 0
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "doub_exp_syn"
        self.__class__.n += 1
        
class IF_cond_alpha(common.IF_cond_alpha, IF_base):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    n = 0
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "alpha_syn"
        self.__class__.n += 1

class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        raise Exception('Cell type %s not yet implemented' % self.__class__.__name__)

class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        raise Exception('Cell type %s not yet implemented' % self.__class__.__name__)


# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global xmldoc, xmlfile, populations_node, projections_node, inputs_node, cells_node, channels_node
    xmlfile = extra_params['file']
    dt = timestep
    xmldoc = xml.dom.minidom.Document()
    neuromlNode = xmldoc.createElementNS(neuroml_url+'/neuroml/schema','neuroml')
    neuromlNode.setAttributeNS(namespace['xsi'],'xsi:schemaLocation',"http://morphml.org/neuroml/schema ../../Schemata/v1.5/Level3/NeuroML_Level3_v1.5.xsd")
    neuromlNode.setAttribute('lengthUnits',"micron")
    xmldoc.appendChild(neuromlNode)
    
    populations_node = build_node('net:populations')
    projections_node = build_node('net:projections', units="Physiological Units")
    inputs_node = build_node('net:inputs', units="Physiological Units")
    cells_node = build_node(':cells')
    channels_node = build_node(':channels')
    
    for node in populations_node, projections_node, inputs_node, cells_node, channels_node:
        neuromlNode.appendChild(node)
        
def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    xml.dom.ext.PrettyPrint(xmldoc, xmlfile)

def run(simtime):
    """Run the simulation for simtime ms."""
    pass

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    raise Exception('Not yet implemented')

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass,paramDict=None,n=1):
    """Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    raise Exception('Not yet implemented')

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or uS."""
    raise Exception('Not yet implemented')

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    raise Exception('Not yet implemented')

def record(source,filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    pass # put a comment in the NeuroML file?

def record_v(source,filename):
    """Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    pass # put a comment in the NeuroML file?

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================
    
class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    
    n = 0
    
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
        global populations_node, cells_node, channels_node
        common.Population.__init__(self,dims,cellclass,cellparams,label)
        self.label = self.label or 'Population%d' % Population.n
        self.celltype = cellclass(cellparams)
        Population.n += 1
        
        population_node = build_node('net:population', name=self.label)
        self.celltype.label = '%s_%s' % (self.celltype.__class__.__name__, self.label)
        celltype_node = build_node('net:cell_type', self.celltype.label)
        instances_node = build_node('net:instances')
        for i in range(self.size):
            x,y,z = self.positions[:,i]
            instance_node = build_node('net:instance', id=i)
            instance_node.appendChild( build_node('net:location', x=x, y=y, z=z) )
            instances_node.appendChild(instance_node)
            
        for node in celltype_node, instances_node:
            population_node.appendChild(node)
        
        populations_node.appendChild(population_node)

        cell_node, channel_list = self.celltype.build_nodes()
        cells_node.appendChild(cell_node)
        for channel_node in channel_list:
            channels_node.appendChild(channel_node)
            
class Projection(common.Projection):
    """
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    n = 0
    
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
        global projections_node
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,methodParameters,source,target,label,rng)
        self.label = self.label or 'Projection%d' % Projection.n
        connection_method = getattr(self,'_%s' % method)
        self.synapse_type = target
        
        projection_node = build_node('net:projection', name=self.label)
        projection_node.appendChild( build_node('net:source', self.pre.label) )
        projection_node.appendChild( build_node('net:target', self.post.label) )
        
        projections_node.appendChild(projection_node)
        
        Projection.n += 1