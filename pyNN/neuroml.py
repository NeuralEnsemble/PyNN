# encoding: utf-8
"""
PyNN-->NeuroML

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import common, connectors, cells, standardmodels
import math
import numpy
import sys
sys.path.append('/usr/lib/python%s/site-packages/oldxml' % sys.version[:3]) # needed for Ubuntu
import xml.dom.ext
import xml.dom.minidom

neuroml_url = 'http://morphml.org'
namespace = {'xsi': "http://www.w3.org/2001/XMLSchema-instance",
             'mml':  neuroml_url+"/morphml/schema",
             'net':  neuroml_url+"/networkml/schema",
             'meta': neuroml_url+"/metadata/schema",
             'bio':  neuroml_url+"/biophysics/schema",  
             'cml':  neuroml_url+"/channelml/schema",}
             
neuroml_ver="1.7.3"
neuroml_xsd="http://www.neuroml.org/NeuroMLValidator/NeuroMLFiles/Schemata/v"+neuroml_ver+"/Level3/NeuroML_Level3_v"+neuroml_ver+".xsd"

strict = False

# ==============================================================================
#   Utility classes
# ==============================================================================

class ID(int, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def __init__(self, n):
        common.IDMixin.__init__(self)

# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

def build_node(name_, text=None, **attributes):
    # we call the node name 'name_' because 'name' is a common attribute name (confused? I am)
    ns, name_ = name_.split(':')
    if ns:
        node = xmldoc.createElementNS(namespace[ns], "%s:%s" % (ns, name_))
    else:
        node = xmldoc.createElement(name_)
    for attr, value in attributes.items():
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
        soma_node = build_node('mml:segment', id=0, name="Soma", cable=0)
        # L = 100  diam = 1000/PI: gives area = 10³ cm²
        soma_node.appendChild(build_node('mml:proximal', x=0, y=0, z=0, diameter=1000/math.pi))
        soma_node.appendChild(build_node('mml:distal', x=0, y=0, z=100, diameter=1000/math.pi))
        segments_node.appendChild(soma_node)
        
        cables_node   = build_node('mml:cables')
        soma_node = build_node('mml:cable', id=0, name="Soma")
        soma_node.appendChild(build_node('meta:group','all'))
        cables_node.appendChild(soma_node)
        return segments_node, cables_node
        
    def define_biophysics(self):
        # L = 100  diam = 1000/PI  // 
        biophys_node  = build_node(':biophysics', units="Physiological Units")
        ifnode        = build_node('bio:mechanism', name="IandF_"+self.label, type='Channel Mechanism')
        passive_node  = build_node('bio:mechanism', name="pas_"+self.label, type='Channel Mechanism', passive_conductance="true")
        # g_max = 10�?�³cm/tau_m  // cm(nF)/tau_m(ms) = G(µS) = 10�?��?�G(S). Divide by area (10³) to get factor of 10�?�³
        gmax = str(1e-3*self.parameters['cm']/self.parameters['tau_m'])
        passive_node.appendChild(build_parameter_node('gmax', gmax))
        cm_node       = build_node('bio:specificCapacitance')
        cm_node.appendChild(build_parameter_node('', str(self.parameters['cm'])))  # units?
        Ra_node       = build_node('bio:specificAxialResistance')
        Ra_node.appendChild(build_parameter_node('', "0.1")) # value doesn't matter for a single compartment
        # These are not needed here
        #esyn_node     = build_node('bio:mechanism', name="ExcitatorySynapse", type="Channel Mechanism")
        #isyn_node     = build_node('bio:mechanism', name="InhibitorySynapse", type="Channel Mechanism")
        
        for node in ifnode, passive_node, cm_node, Ra_node: # the order is important here
            biophys_node.appendChild(node)
        return biophys_node
        
    def define_connectivity(self):
        conn_node  = build_node(':connectivity')
        esyn_node  = build_node('net:potential_syn_loc', synapse_type="ExcSyn_"+self.label, synapse_direction="preAndOrPost")
        esyn_node.appendChild(build_node('net:group'))
        isyn_node  = build_node('net:potential_syn_loc', synapse_type="InhSyn_"+self.label, synapse_direction="preAndOrPost")
        isyn_node.appendChild(build_node('net:group'))
        
      
        for node in esyn_node, isyn_node: 
            conn_node.appendChild(node)
        return conn_node
        
    def define_channel_types(self):
        
        passive_node = build_node('cml:channel_type', name="pas_"+self.label, density="yes")
        passive_node.appendChild( build_node('meta:notes', "Simple example of a leak/passive conductance") )
        gmax = str(1e-3*self.parameters['cm']/self.parameters['tau_m'])
        
        cvr_node = build_node('cml:current_voltage_relation', 
                              cond_law="ohmic",
                              ion="non_specific",
                              default_gmax=gmax,
                              default_erev=self.parameters['v_rest'])
                              
        passive_node.appendChild(cvr_node)
        
        ifnode = build_node('cml:channel_type', name="IandF_"+self.label)
        ifnode.appendChild( build_node('meta:notes', "Spike and reset mechanism") )
        cvr_node = build_node('cml:current_voltage_relation')
        ifmech_node = build_node('cml:integrate_and_fire',
                                 threshold=self.parameters['v_thresh'],
                                 t_refrac=self.parameters['tau_refrac'],
                                 v_reset=self.parameters['v_reset'],
                                 g_refrac=0.1) # this value just needs to be 'large enough'
        cvr_node.appendChild(ifmech_node)
        ifnode.appendChild(cvr_node)
        
        return [passive_node, ifnode]
            
    def define_synapse_types(self, synapse_type):
        esyn_node = build_node('cml:synapse_type', name="ExcSyn_"+self.label)
        rise_time_exc="0"
        rise_time_inh="0"
        
        if (synapse_type == 'alpha_syn'):
            rise_time_exc=self.parameters['tau_syn_E']
            rise_time_inh=self.parameters['tau_syn_I']
            
        esyn_node.appendChild( build_node('cml:doub_exp_syn',
                                          max_conductance="1.0e-5",
                                          rise_time=rise_time_exc,
                                          decay_time=self.parameters['tau_syn_E'],
                                          reversal_potential=self.parameters['e_rev_E'] ) )
                                          
        isyn_node = build_node('cml:synapse_type', name="InhSynSyn_"+self.label)
        isyn_node.appendChild( build_node('cml:doub_exp_syn',
                                          max_conductance="1.0e-5",
                                          rise_time=rise_time_inh,
                                          decay_time=self.parameters['tau_syn_I'],
                                          reversal_potential=self.parameters['e_rev_I'] ) )
        return [esyn_node, isyn_node]

    def build_nodes(self):
        cell_node = build_node(':cell', name=self.label)
        doc_node = build_node('meta:notes', "Instance of PyNN %s cell type" % self.__class__.__name__)
        segments_node, cables_node = self.define_morphology()
        biophys_node = self.define_biophysics()
        conn_node = self.define_connectivity()
        
        for node in doc_node, segments_node, cables_node, biophys_node, conn_node:
            cell_node.appendChild(node)
        
        channel_nodes = self.define_channel_types()
        synapse_nodes = self.define_synapse_types(self.synapse_type)
        
        return cell_node, channel_nodes, synapse_nodes


class NotImplementedModel(object):
    
    def __init__(self):
        if strict:
            raise Exception('Cell type %s is not available in NeuroML' % self.__class__.__name__)
    
    def build_nodes(self):
        cell_node = build_node(':not_implemented_cell', name=self.label)
        doc_node = build_node('meta:notes', "PyNN %s cell type not implemented" % self.__class__.__name__)
        return cell_node, [], []
        

# ==============================================================================
#   Standard cells
# ==============================================================================

class IF_curr_exp(cells.IF_curr_exp, NotImplementedModel):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses"""
    
    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.IF_curr_exp.default_parameters])
    
    def __init__(self, parameters):
        NotImplementedModel.__init__(self)
        cells.IF_curr_exp.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "doub_exp_syn"
        self.__class__.n += 1

class IF_curr_alpha(cells.IF_curr_alpha, NotImplementedModel):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.IF_curr_alpha.default_parameters])
    
    def __init__(self, parameters):
        NotImplementedModel.__init__(self)
        cells.IF_curr_exp.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "doub_exp_syn"
        self.__class__.n += 1

class IF_cond_exp(cells.IF_cond_exp, IF_base):
    """Leaky integrate and fire model with fixed threshold and 
    decaying-exponential post-synaptic conductance."""
    
    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.IF_cond_exp.default_parameters])
    
    def __init__(self, parameters):
        cells.IF_cond_exp.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "doub_exp_syn"
        self.__class__.n += 1
        
class IF_cond_alpha(cells.IF_cond_alpha, IF_base):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.IF_cond_alpha.default_parameters])
    
    def __init__(self, parameters):
        cells.IF_cond_alpha.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "alpha_syn"
        self.__class__.n += 1

class SpikeSourcePoisson(cells.SpikeSourcePoisson, NotImplementedModel):
    """Spike source, generating spikes according to a Poisson process."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.SpikeSourcePoisson.default_parameters])
    
    def __init__(self, parameters):
        NotImplementedModel.__init__(self)
        cells.SpikeSourcePoisson.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.__class__.n += 1
        

class SpikeSourceArray(cells.SpikeSourceArray, NotImplementedModel):
    """Spike source generating spikes at the times given in the spike_times array."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.SpikeSourcePoisson.default_parameters])

    def __init__(self, parameters):
        NotImplementedModel.__init__(self)
        cells.SpikeSourceARRAY.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.__class__.n += 1
        


# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=0.1, debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global xmldoc, xmlfile, populations_node, projections_node, inputs_node, cells_node, channels_node, neuromlNode, strict
    xmlfile = extra_params['file']
    if isinstance(xmlfile, basestring):
        xmlfile = open(xmlfile, 'w')
    if 'strict' in extra_params:
        strict = extra_params['strict']
    dt = timestep
    xmldoc = xml.dom.minidom.Document()
    neuromlNode = xmldoc.createElementNS(neuroml_url+'/neuroml/schema','neuroml')
    neuromlNode.setAttributeNS(namespace['xsi'],'xsi:schemaLocation',"http://morphml.org/neuroml/schema "+neuroml_xsd)
    neuromlNode.setAttribute('lengthUnits',"micron")
    xmldoc.appendChild(neuromlNode)
    
    populations_node = build_node('net:populations')
    projections_node = build_node('net:projections', units="Physiological Units")
    inputs_node = build_node('net:inputs', units="Physiological Units")
    cells_node = build_node(':cells')
    channels_node = build_node(':channels', units="Physiological Units")
    
    for node in cells_node, channels_node, populations_node, projections_node, inputs_node:
        neuromlNode.appendChild(node)
    return 0
        
def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global xmldoc, xmlfile, populations_node, projections_node, inputs_node, cells_node, channels_node, neuromlNode
    # Remove empty nodes, otherwise the validator will complain
    for node in cells_node, channels_node, populations_node, projections_node, inputs_node:
        if not node.hasChildNodes():
            neuromlNode.removeChild(node)
    # Write the file
    xml.dom.ext.PrettyPrint(xmldoc, xmlfile)
    xmlfile.close()

def run(simtime):
    """Run the simulation for simtime ms."""
    pass # comment in NeuroML file


def get_min_delay():
    return 0.0
common.get_min_delay = get_min_delay

def num_processes():
    return 1
common.num_processes = num_processes

def rank():
    return 0
common.rank = rank


# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, cellparams=None, n=1):
    """Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    raise Exception('Not yet implemented')

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or uS."""
    raise Exception('Not yet implemented')

def set(cells, cellclass, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    raise Exception('Not yet implemented')

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    pass # put a comment in the NeuroML file?

def record_v(source, filename):
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
    
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 label=None):
        """
        Create a population of neurons all of the same type.
        
        size - number of cells in the Population. For backwards-compatibility, n
               may also be a tuple giving the dimensions of a grid, e.g. n=(10,10)
               is equivalent to n=100 with structure=Grid2D()
        cellclass should either be a standardized cell class (a class inheriting
        from common.standardmodels.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        structure should be a Structure instance.
        label is an optional name for the population.
        """
        global populations_node, cells_node, channels_node
        common.Population.__init__(self, size, cellclass, cellparams, structure, label)
        self.label = self.label or 'Population%d' % Population.n
        self.celltype = cellclass(cellparams)
        Population.n += 1
        
        population_node = build_node('net:population', name=self.label)
        self.celltype.label = '%s_%s' % (self.celltype.__class__.__name__, self.label)
        celltype_node = build_node('net:cell_type', self.celltype.label)
        instances_node = build_node('net:instances', size=self.size)
        for i in range(self.size):
            x, y, z = self.positions[:, i]
            instance_node = build_node('net:instance', id=i)
            instance_node.appendChild( build_node('net:location', x=x, y=y, z=z) )
            instances_node.appendChild(instance_node)
            
        for node in celltype_node, instances_node:
            population_node.appendChild(node)
        
        populations_node.appendChild(population_node)

        cell_node, channel_list, synapse_list = self.celltype.build_nodes()
        cells_node.appendChild(cell_node)
        # Add all channels first, then all synapses
        for channel_node in channel_list:
            channels_node.insertBefore(channel_node , channels_node.firstChild)
        for synapse_node in synapse_list:
            channels_node.appendChild(synapse_node)

        self.first_id = 0
        self.last_id = self.size-1
        self.all_cells = numpy.array([ID(id) for id in range(self.first_id, self.last_id+1)], dtype=ID)
        self._mask_local = numpy.ones_like(self.all_cells).astype(bool)
        self.local_cells = self.all_cells[self._mask_local]

    def _record(self, variable, record_from=None, rng=None, to_file=True):
        """
        Private method called by record() and record_v().
        """
        pass
    
    def mean_spike_count(self):
        return -1
    
    def printSpikes(self, file, gather=True, compatible_output=True):
        pass
    
    def print_v(self, file, gather=True, compatible_output=True):
        pass

class AllToAllConnector(connectors.AllToAllConnector):
    
    def connect(self, projection):
        connectivity_node = build_node('net:connectivity_pattern')
        connectivity_node.appendChild( build_node('net:all_to_all',
                                                  allow_self_connections=int(self.allow_self_connections)) )
        return connectivity_node

class OneToOneConnector(connectors.OneToOneConnector):
    
    def connect(self, projection):
        connectivity_node = build_node('net:connectivity_pattern')
        connectivity_node.appendChild( build_node('net:one_to_one') )
        return connectivity_node

class FixedProbabilityConnector(connectors.FixedProbabilityConnector):
    
    def connect(self, projection):
        connectivity_node = build_node('net:connectivity_pattern')
        connectivity_node.appendChild( build_node('net:fixed_probability',
                                                  probability=self.p_connect,
                                                  allow_self_conections=int(self.allow_self_connections)) )
        return connectivity_node


class FixedNumberPreConnector(connectors.FixedNumberPreConnector):
    
    def connect(self, projection):
        if hasattr(self, "n"):
            connectivity_node = build_node('net:connectivity_pattern')
            connectivity_node.appendChild( build_node('net:per_cell_connection',
                                                      num_per_source=self.n,
                                                      direction="PreToPost",
                                                      allow_self_connections = int(self.allow_self_connections)) )
            return connectivity_node
        else:
            raise Exception('Connection with variable connection number not implemented.')
    
class FixedNumberPostConnector(connectors.FixedNumberPostConnector):
    
    def connect(self, projection):
        if hasattr(self, "n"):
            connectivity_node = build_node('net:connectivity_pattern')
            connectivity_node.appendChild( build_node('net:per_cell_connection',
                                                      num_per_source=self.n,
                                                      direction="PostToPre",
                                                      allow_self_connections = int(self.allow_self_connections)) )
            return connectivity_node
        else:
            raise Exception('Connection with variable connection number not implemented.')

        
class FromListConnector(connectors.FromListConnector):
    
    def connect(self, projection):
        connections_node = build_node('net:connections')
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = self.pre[tuple(src)]
            tgt = self.post[tuple(tgt)]
            connection_node = build_node('net:connection', id=i)
            connection_node.appendChild( build_node('net:pre', cell_id=src) )
            connection_node.appendChild( build_node('net:post', cell_id=tgt) )
            connection_node.appendChild( build_node('net:properties', internal_delay=delay, weight=weight) )
            connections_node.appendChild(connection_node)
        return connections_node


class FromFileConnector(connectors.FromFileConnector):
    
    def connect(self, projection):
        # now open the file...
        f = open(self.filename,'r',10000)
        lines = f.readlines()
        f.close()
        
        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            input_tuples.append((eval(src), eval(tgt), float(w), float(d)))
        f.close()
        self.conn_list = input_tuples
        FromListConnector.connect(projection)


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    n = 0
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 method,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell signals action potentials
        
        target - string specifying which synapse on the postsynaptic cell to connect to
        If source and/or target are not given, default values are used.
        
        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.
        
        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.
        
        rng - specify an RNG object to be used by the Connector.
        """
        global projections_node
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target, synapse_dynamics, label, rng)
        self.label = self.label or 'Projection%d' % Projection.n
        connection_method = method
        if target:
            self.synapse_type = target
        else:
            self.synapse_type = "ExcitatorySynapse"
        
        projection_node = build_node('net:projection', name=self.label)
        projection_node.appendChild( build_node('net:source', self.pre.label) )
        projection_node.appendChild( build_node('net:target', self.post.label) )
        
        synapse_node = build_node('net:synapse_props')
        synapse_node.appendChild( build_node('net:synapse_type', self.synapse_type) )
        synapse_node.appendChild( build_node('net:default_values', internal_delay=5, weight=1, threshold=-20) )
        projection_node.appendChild(synapse_node)
        
        projection_node.appendChild( connection_method.connect(self) )
        
        projections_node.appendChild(projection_node)
        Projection.n += 1

    def saveConnections(self, filename, gather=True, compatible_output=True):
        pass
    
    def __len__(self):
        return 0 # needs implementing properly

# ==============================================================================
