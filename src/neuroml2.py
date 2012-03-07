"""
PyNN-->NeuroML v2

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

This file is based on neuroml.py written by Andrew Davison & has been updated for
NeuuroML v2.0 by Padraig Gleeson

"""

'''

This script is intended to map PyNN scripts on to the equivalent representation in
NeuroML v2.0. A valid NML2 file will be produced containing the cells, populations,
etc. and a LEMS file will be created which imports this file and can run a simple
simulation using the LEMS interpreter, see http://www.neuroml.org/lems/interpreter.html

Ideally... this will produce equivalent simulation results when a script is run using:
    python myPyNN.py nest
    python myPyNN.py neuron
    python myPyNN.py neuroml2 (followed by nml2 LEMS_PyNN2NeuroMLv2.xml)

        WORK IN PROGRESS! REQUIRES PyNN at tags/0.7.2/

To test this out get the full PyNN tree from SVN using: svn co https://neuralensemble.org/svn/PyNN/
then go to tags/0.7.2/src, copy neuroml2.py there, and install using setup.py in tags/0.7.2

Contact p.gleeson@ucl.ac.uk for more details 

Features below depend on using the latest LEMS/libNeuroML code which includes the
nml2 utility and the LEMS definitions of PyNN core models (IF_curr_alpha,
SpikeSourcePoisson, etc.) in PyNN.xml. Get it from
http://sourceforge.net/apps/trac/neuroml/browser/NeuroML2/


Currently supported features:
    Generation of valid NeuroML 2 file containing cells & populations & connections
    Export of simulation duration & dt & recorded populations in a LEMS file for
       running a basic simulation with simple num integration method (so use small dt!)
    Cell models impl: IF_curr_alpha, IF_curr_exp, IF_cond_exp, IF_cond_alpha, HH_cond_exp, EIF_cond_exp_isfa_ista, EIF_cond_alpha_isfa_ista
    Others: SpikeSourcePoisson, SpikeSourceArray
    Export of explicitly created Populations, export of populations created with create()
    Export of (instance based) list of conenctions in explicit <connection from=... to=...>
    Support for weight & delay on connections

Missing/required:
    Other models todo: DCSource, StepCurrentSource, ACSource, NoisyCurrentSource
    Need to test >1 cells in a population
    Setting of initial values in Populations
    Support for populations some of whose cells have has their parameters modified
    Synapse dynamics (e.g. STDP) not yet implemented


Desirable TODO:
    Generation of SED-ML file with simulation description
    Automated tests of equivalence between Neuron & Nest & generated LEMS

'''

from pyNN import common, connectors, standardmodels, core
from pyNN.standardmodels import cells

import numpy
import sys

sys.path.append('/usr/lib/python%s/site-packages/oldxml' % sys.version[:3]) # needed for Ubuntu
import xml.dom.minidom

import logging
logger = logging.getLogger("neuroml2")

neuroml_ns = 'http://www.neuroml.org/schema/neuroml2'

namespace_xsi = "http://www.w3.org/2001/XMLSchema-instance"

neuroml_ver="v2alpha"
neuroml_xsd="http://neuroml.svn.sourceforge.net/viewvc/neuroml/NeuroML2/Schemas/NeuroML2/NeuroML_"+neuroml_ver+".xsd"

simulation_prefix = 'simulation_'
network_prefix = 'network_'
display_prefix = 'display_'
line_prefix = 'line_'
colours = ['#000000','#FF0000','#0000FF','#009b00','#ffc800','#8c6400','#ff00ff','#ffff00','#808080']

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


    def get_native_parameters(self):
        """Return a dictionary of parameters for the NeuroML2 cell model."""
     
        return self._cell

    def set_native_parameters(self, parameters):
        """Set parameters of the NeuroML2 cell model from a dictionary.
        for name, val in parameters.items():
            setattr(self._cell, name, val)"""
        self._cell =    parameters.copy()

# ==============================================================================
#   Module-specific functions and classes (not part of the common API)
# ==============================================================================

def build_node(name_, text=None, **attributes):
    # we call the node name 'name_' because 'name' is a common attribute name (confused? I am)

    node = nml2doc.createElement(name_)
    for attr, value in attributes.items():
        node.setAttribute(attr, str(value))
    if text:
        node.appendChild(nml2doc.createTextNode(text))
    return node

def build_parameter_node(name, value):
        param_node = build_node('parameter', value=value)
        if name:
            param_node.setAttribute('name', name)
        group_node = build_node('group', 'all')
        param_node.appendChild(group_node)
        return param_node


class IF_base(object):
    """Base class for integrate-and-fire neuron models."""        


    def build_nodes(self):
        cell_type = self.__class__.__name__
        logger.debug("Building nodes for "+cell_type)

        #cell_node = build_node('component', type=self.__class__.__name__, id=self.label)
        cell_node = build_node(cell_type, id=self.label)
        
        for param in self.parameters.keys():
            paral_val = str(self.parameters[param])

            # TODO why is this broken for a in EIF_cond_exp_isfa_ista????
            if "EIF_cond_" in cell_type and param is "a":
                paral_val = float(paral_val)
                paral_val = paral_val/1000.
                
            logger.debug("Setting param %s to %s"%(param, paral_val))
            
            cell_node.setAttribute(param, str(paral_val))

        ##TODO remove!!
        cell_node.setAttribute('v_init', '-65')
            
        doc_node = build_node('notes', "Component for PyNN %s cell type" % cell_type)
        cell_node.appendChild(doc_node)

        synapse_nodes = []
        if 'cond_exp' in cell_type:
            synapse_nodes_e = build_node("expCondSynapse", id="syn_e_"+self.label)
            synapse_nodes_e.setAttribute("tau_syn",str(self.parameters["tau_syn_E"]))
            synapse_nodes_e.setAttribute("e_rev",str(self.parameters["e_rev_E"]))
            synapse_nodes.append(synapse_nodes_e)
            synapse_nodes_i = build_node("expCondSynapse", id="syn_i_"+self.label)
            synapse_nodes_i.setAttribute("tau_syn",str(self.parameters["tau_syn_I"]))
            synapse_nodes_i.setAttribute("e_rev",str(self.parameters["e_rev_I"]))
            synapse_nodes.append(synapse_nodes_i)
        elif 'cond_alpha' in cell_type:
            synapse_nodes_e = build_node("alphaCondSynapse", id="syn_e_"+self.label)
            synapse_nodes_e.setAttribute("tau_syn",str(self.parameters["tau_syn_E"]))
            synapse_nodes_e.setAttribute("e_rev",str(self.parameters["e_rev_E"]))
            synapse_nodes.append(synapse_nodes_e)
            synapse_nodes_i = build_node("alphaCondSynapse", id="syn_i_"+self.label)
            synapse_nodes_i.setAttribute("tau_syn",str(self.parameters["tau_syn_I"]))
            synapse_nodes_i.setAttribute("e_rev",str(self.parameters["e_rev_I"]))
            synapse_nodes.append(synapse_nodes_i)
        elif 'curr_exp' in cell_type:
            synapse_nodes_e = build_node("expCurrSynapse", id="syn_e_"+self.label)
            synapse_nodes_e.setAttribute("tau_syn",str(self.parameters["tau_syn_E"]))
            synapse_nodes.append(synapse_nodes_e)
            synapse_nodes_i = build_node("expCurrSynapse", id="syn_i_"+self.label)
            synapse_nodes_i.setAttribute("tau_syn",str(self.parameters["tau_syn_I"]))
            synapse_nodes.append(synapse_nodes_i)
        elif 'curr_alpha' in cell_type:
            synapse_nodes_e = build_node("alphaCurrSynapse", id="syn_e_"+self.label)
            synapse_nodes_e.setAttribute("tau_syn",str(self.parameters["tau_syn_E"]))
            synapse_nodes.append(synapse_nodes_e)
            synapse_nodes_i = build_node("alphaCurrSynapse", id="syn_i_"+self.label)
            synapse_nodes_i.setAttribute("tau_syn",str(self.parameters["tau_syn_I"]))
            synapse_nodes.append(synapse_nodes_i)

        
        return cell_node, synapse_nodes


class NotImplementedModel(object):
    
    def __init__(self):
        if strict:
            raise Exception('Cell type %s is not available in NeuroML' % self.__class__.__name__)
    
    def build_nodes(self):
        cell_node = build_node(':not_implemented_cell', id=self.label)
        doc_node = build_node('notes', "PyNN %s cell type not implemented" % self.__class__.__name__)
        cell_node.appendChild(doc_node)
        return cell_node, []
        

# ==============================================================================
#   Standard cells
# ==============================================================================

class IF_curr_exp(cells.IF_curr_exp, IF_base):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses"""
    
    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.IF_curr_exp.default_parameters])
    
    def __init__(self, parameters):
        cells.IF_curr_exp.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "doub_exp_syn"
        self.__class__.n += 1
        logger.debug("IF_curr_exp created")


class IF_curr_alpha(cells.IF_curr_alpha, IF_base):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.IF_curr_alpha.default_parameters])
    
    def __init__(self, parameters):
        cells.IF_curr_alpha.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "doub_exp_syn"
        self.__class__.n += 1
        logger.debug("IF_curr_alpha created")


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
        logger.debug("IF_cond_exp created")


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
        logger.debug("IF_cond_alpha created")


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista, IF_base):
    """Exponential integrate and fire neuron with spike triggered and sub-threshold adaptation currents (isfa, ista reps.) according to:
Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.EIF_cond_exp_isfa_ista.default_parameters])

    def __init__(self, parameters):
        cells.EIF_cond_exp_isfa_ista.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "exp_syn"
        self.__class__.n += 1
        logger.debug("EIF_cond_exp_isfa_ista created")


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista, IF_base):
    """Exponential integrate and fire neuron with spike triggered and sub-threshold adaptation currents (isfa, ista reps.) according to:
Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.EIF_cond_alpha_isfa_ista.default_parameters])

    def __init__(self, parameters):
        cells.EIF_cond_alpha_isfa_ista.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "alpha_syn"
        self.__class__.n += 1
        logger.debug("EIF_cond_alpha_isfa_ista created")


class HH_cond_exp(cells.HH_cond_exp, IF_base):
    """ Single-compartment Hodgkin-Huxley model."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.HH_cond_exp.default_parameters])

    def __init__(self, parameters):
        cells.HH_cond_exp.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.synapse_type = "exp_syn"
        self.__class__.n += 1
        logger.debug("HH_cond_exp created")


class GenericModel(object):

    units_to_use = {}

    def build_nodes(self):
        logger.debug("Building nodes for "+self.__class__.__name__)

        model_node = build_node(self.__class__.__name__, id=self.label)

        for param in self.parameters.keys():
            units = ''
            if param in self.units_to_use.keys():
                units = self.units_to_use[param]
            model_node.setAttribute(param, str(self.parameters[param])+units)


        doc_node = build_node('notes', "Component for PyNN %s model type" % self.__class__.__name__)
        model_node.appendChild(doc_node)

        return model_node, []


class SpikeSourcePoisson(cells.SpikeSourcePoisson, GenericModel):
    """Spike source, generating spikes according to a Poisson process."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.SpikeSourcePoisson.default_parameters])


    def __init__(self, parameters):
        cells.SpikeSourcePoisson.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.__class__.n += 1
        self.units_to_use = {'start':'ms','duration':'ms','rate':'per_s'}
        logger.debug("SpikeSourcePoisson created: "+self.label)
        

class SpikeSourceArray(cells.SpikeSourceArray, GenericModel):
    """Spike source generating spikes at the times given in the spike_times array."""

    n = 0
    translations = standardmodels.build_translations(*[(name, name)
                                               for name in cells.SpikeSourceArray.default_parameters])

    def __init__(self, parameters):
        cells.SpikeSourceArray.__init__(self, parameters)
        self.label = '%s%d' % (self.__class__.__name__, self.__class__.n)
        self.__class__.n += 1
        logger.debug("SpikeSourceArray created: "+self.label)

    def build_nodes(self):
        logger.debug("Building nodes for "+self.__class__.__name__)

        model_node = build_node('spikeArray', id=self.label)
        #doc_node = build_node('notes', "Component for PyNN %s model type" % self.__class__.__name__)
        #model_node.appendChild(doc_node)

        for spike in self.parameters['spike_times']:
            spike_node = build_node('spike', time="%fms"%spike)
            model_node.appendChild(spike_node)

        return model_node, []


# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=0.1, debug=False,**extra_params):

    logger.debug("setup() called, extra_params = "+str(extra_params))
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global nml2doc, nml2file, lemsdoc, lemsfile, lemsNode, nml_id, population_holder, projection_holder, input_holder, cell_holder, channel_holder, neuromlNode, strict, dt

    population_holder = []
    projection_holder = []
    input_holder = []
    cell_holder = []
    
    if not extra_params.has_key('file'):
        nml2file = "PyNN2NeuroMLv2.nml"
    else:
        nml2file = extra_params['file']

    nml_id = nml2file.split('.')[0]

    if isinstance(nml2file, basestring):
        nml2file = open(nml2file, 'w')

    if 'strict' in extra_params:
        strict = extra_params['strict']
    dt = timestep

    nml2doc = xml.dom.minidom.Document()
    neuromlNode = nml2doc.createElementNS(neuroml_ns,'neuroml')
    neuromlNode.setAttribute("xmlns",neuroml_ns)

    neuromlNode.setAttribute('xmlns:xsi',namespace_xsi)
    neuromlNode.setAttribute('xsi:schemaLocation',neuroml_ns+" "+neuroml_xsd)
    neuromlNode.setAttribute('id',nml_id)


    nml2doc.appendChild(neuromlNode)
    

    lemsdoc = xml.dom.minidom.Document()
    lemsNode = lemsdoc.createElement('Lems')
    lemsdoc.appendChild(lemsNode)

    drNode = build_node('DefaultRun',component=simulation_prefix+nml_id)
    lemsNode.appendChild(drNode)
    coreNml2Files = ["NeuroMLCoreDimensions.xml","PyNN.xml","Networks.xml","Simulation.xml"]
    for f in coreNml2Files:
        incNode = build_node('Include', file="NeuroML2CoreTypes/"+f)
        lemsNode.appendChild(incNode)

    incNode = build_node('Include', file=nml2file.name)
    lemsNode.appendChild(incNode)

    global simNode, displayNode
    simNode = build_node('Simulation', id=simulation_prefix+nml_id, step=str(dt)+"ms", target=network_prefix+nml_id)
    lemsNode.appendChild(simNode)
    displayNode = build_node('Display',id="display_0",title="Recording of PyNN model run in LEMS", timeScale="1ms")
    simNode.appendChild(displayNode)

    lemsfile = "LEMS_"+nml_id+".xml"
    if isinstance(lemsfile, basestring):
        lemsfile = open(lemsfile, 'w')
        
    return 0
        
def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global nml2doc, nml2file, neuromlNode, nml_id


    for cellNode in cell_holder:
        neuromlNode.appendChild(cellNode)

  
    network_node = build_node('network', id=network_prefix+nml_id)
    neuromlNode.appendChild(network_node)

    for holder in population_holder, projection_holder, input_holder:
        for node in holder:
            network_node.appendChild(node)

    # Write the files
    logger.debug("Writing NeuroML 2 structure to: "+nml2file.name)
    nml2file.write(nml2doc.toprettyxml())
    nml2file.close()

    logger.debug("Writing LEMS file to: "+lemsfile.name)
    lemsfile.write(lemsdoc.toprettyxml())
    lemsfile.close()
    print("\nThe file: "+lemsfile.name+" has been generated. This can be executed with libNeuroML utility nml2 (which wraps the LEMS Interpreter), i.e.")
    print("\n    nml2 "+lemsfile.name+"\n")


def run(simtime):
    """Run the simulation for simtime ms."""
    global simNode
    simNode.setAttribute('length', str(simtime)+"ms")



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
        __doc__ = common.Population.__doc__
        common.Population.__init__(self, size, cellclass, cellparams, structure, label)
        ###simulator.initializer.register(self)

    def _create_cells(self, cellclass, cellparams, n):
        """
        Create a population of neurons all of the same type.
        

        `cellclass`  -- a PyNN standard cell
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        global population_holder, cell_holder, channel_holder

        assert n > 0, 'n must be a positive integer'

        self.celltype = cellclass(cellparams)
        Population.n += 1

        self.celltype.label = 'cell_%s' % (self.label)

        population_node = build_node('population', id=self.label, component=self.celltype.label, size=self.size)

        #celltype_node = build_node('cell_type', self.celltype.label)

        instances_node = build_node('instances', size=self.size)
        for i in range(self.size):
            x, y, z = self.positions[:, i]
            instance_node = build_node('instance', id=i)
            instance_node.appendChild( build_node('location', x=x, y=y, z=z) )
            instances_node.appendChild(instance_node)
            
        #population_node.appendChild(node)
        
        population_holder.append(population_node)

        cell_node, synapse_nodes = self.celltype.build_nodes()
        cell_holder.append(cell_node)
        for syn_node in synapse_nodes:
            cell_holder.append(syn_node)


        # Add all channels first, then all synapses
        '''
        for channel_node in channel_list:
            channel_holder_node.insertBefore(channel_node , channel_holder_node.firstChild)
        for synapse_node in synapse_list:
            channel_holder_node.appendChild(synapse_node)'''

        self.first_id = 0
        self.last_id = self.size-1
        self.all_cells = numpy.array([ID(id) for id in range(self.first_id, self.last_id+1)], dtype=ID)
        self._mask_local = numpy.ones_like(self.all_cells).astype(bool)
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        for id in self.all_cells:
            id.parent = self
            id._cell = self.celltype.parameters.copy()
        
        #self.local_cells = self.all_cells


    def _set_initial_value_array(self, variable, value):
        logger.debug("Population %s having %s initialised to: %s"%(self.label, variable, value))

        # TODO: use this in generated XML for component...
        if variable is 'v':
            self.celltype.parameters['v_init'] = value

        
    def _record(self, variable, record_from=None, rng=None, to_file=True):
        """
        Private method called by record() and record_v().
        """
        global simNode, displayNode, color
        #displayNode = build_node('Display',id=display_prefix+self.label,title="Recording of "+variable+" in "+self.label, timeScale="1ms")
        #simNode.appendChild(displayNode)

        scale = "1"
        #if variable == 'v': scale = "1mV"
        colour = colours[displayNode.childNodes.length%len(colours)]
        for i in range(self.size):
            lineNode = build_node('Line',
                                  id=line_prefix+self.label,
                                  scale=scale,
                                  color=colour,
                                  quantity="%s[%i]/%s"%(self.label,i,variable),
                                  save="%s_%i_%s_nml2.dat"%(self.label,i,variable))
                                  
            displayNode.appendChild(lineNode)
    
    def meanSpikeCount(self):
        return -1
    
    def printSpikes(self, file, gather=True, compatible_output=True):
        pass
    
    def print_v(self, file, gather=True, compatible_output=True):
        pass
'''
class AllToAllConnector(connectors.AllToAllConnector):
    
    def connect(self, projection):
        connectivity_node = build_node('connectivity_pattern')
        connectivity_node.appendChild( build_node('all_to_all',
                                                  allow_self_connections=int(self.allow_self_connections)) )
        return connectivity_node

class OneToOneConnector(connectors.OneToOneConnector):
    
    def connect(self, projection):
        connectivity_node = build_node('connectivity_pattern')
        connectivity_node.appendChild( build_node('one_to_one') )
        return connectivity_node

class FixedProbabilityConnector(connectors.FixedProbabilityConnector):
    
    def connect(self, projection):
        connectivity_node = build_node('connectivity_pattern')
        connectivity_node.appendChild( build_node('fixed_probability',
                                                  probability=self.p_connect,
                                                  allow_self_conections=int(self.allow_self_connections)) )
        return connectivity_node
'''
FixedProbabilityConnector = connectors.FixedProbabilityConnector
AllToAllConnector = connectors.AllToAllConnector
OneToOneConnector = connectors.OneToOneConnector
CSAConnector = connectors.CSAConnector

class FixedNumberPreConnector(connectors.FixedNumberPreConnector):
    
    def connect(self, projection):
        if hasattr(self, "n"):
            connectivity_node = build_node('connectivity_pattern')
            connectivity_node.appendChild( build_node('per_cell_connection',
                                                      num_per_source=self.n,
                                                      direction="PreToPost",
                                                      allow_self_connections = int(self.allow_self_connections)) )
            return connectivity_node
        else:
            raise Exception('Connection with variable connection number not implemented.')
    
class FixedNumberPostConnector(connectors.FixedNumberPostConnector):
    
    def connect(self, projection):
        if hasattr(self, "n"):
            connectivity_node = build_node('connectivity_pattern')
            connectivity_node.appendChild( build_node('per_cell_connection',
                                                      num_per_source=self.n,
                                                      direction="PostToPre",
                                                      allow_self_connections = int(self.allow_self_connections)) )
            return connectivity_node
        else:
            raise Exception('Connection with variable connection number not implemented.')

        
class FromListConnector(connectors.FromListConnector):
    
    def connect(self, projection):
        connections_node = build_node('connections')
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = self.pre[tuple(src)]
            tgt = self.post[tuple(tgt)]
            connection_node = build_node('connection', id=i)
            connection_node.appendChild( build_node('pre', cell_id=src) )
            connection_node.appendChild( build_node('post', cell_id=tgt) )
            connection_node.appendChild( build_node('properties', internal_delay=delay, weight=weight) )
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
        global projection_holder
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target, synapse_dynamics, label, rng)
        self.label = self.label or 'Projection%d' % Projection.n
        connection_method = method
        if target:
            self.synapse_type = target
        else:
            self.synapse_type = "ExcitatorySynapse"

        synapseComponent = "syn_"

        if self.synapse_type is "ExcitatorySynapse" or self.synapse_type is "excitatory":
            self.targetPort = "spike_in_E"
            synapseComponent = synapseComponent +"e_"
        elif self.synapse_type is "InhibitorySynapse" or self.synapse_type is "inhibitory":
            self.targetPort = "spike_in_I"
            synapseComponent = synapseComponent +"i_"
        else:
            self.targetPort = "spike_in"

        synapseComponent = synapseComponent +"cell_"+postsynaptic_population.label

        self.connection_manager = ConnectionManager(self.synapse_type,
                                                              synapse_model=None,
                                                              parent=self)
        self.connections = self.connection_manager
        ## Create connections
        method.connect(self)

        logger.debug("init in Projection, %s, pre: %s, post %s"%(self.label, presynaptic_population.label, postsynaptic_population.label))
        
        
        #projection_node = build_node('projection', id=self.label)

        for connection in self.connection_manager.connections:
            connection_node = build_node('synapticConnectionWD',
                                                    to='%s[%i]'%(postsynaptic_population.label,connection[1]),
                                                    synapse=synapseComponent)

            connection_node.setAttribute("from",'%s[%i]'%(presynaptic_population.label,connection[0]))
            connection_node.setAttribute("weight",str(connection[3][0]))
            connection_node.setAttribute("delay",str(connection[4][0])+"ms")

            projection_holder.append(connection_node)

        '''
        projection_node.appendChild( build_node('source', self.pre.label) )
        projection_node.appendChild( build_node('target', self.post.label) )
        synapse_node = build_node('synapse_props')
        synapse_node.appendChild( build_node('synapse_type', self.synapse_type) )
        synapse_node.appendChild( build_node('default_values', internal_delay=5, weight=1, threshold=-20) )
        projection_node.appendChild(synapse_node)
        
        projection_node.appendChild( connection_method.connect(self) )
        '''
        projection_holder.append(connection_node)
        Projection.n += 1

    def saveConnections(self, filename, gather=True, compatible_output=True):
        pass
    
    def __len__(self):
        return 0 # needs implementing properly



class ConnectionManager(object):
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.

    Based on ConnectionManager in moose/simulator.py

    """

    def __init__(self, synapse_type, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.

        `parent` -- the parent `Projection`
        """
        assert parent is not None
        self.connections = []
        self.parent = parent
        self.synapse_type = synapse_type
        self.synapse_model = synapse_model

    def connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(source, int) or source < 0:
            errmsg = "Invalid source ID: %s" % (source)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(targets):
            targets = [targets]

        ##############weights = weights*1000.0 # scale units
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        # need to scale weights for appropriate units
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if not isinstance(target, common.IDMixin):
                    raise errors.ConnectionError("Invalid target ID: %s" % target)
                #TODO record weights
                '''
                if self.synapse_type == "excitatory":
                    synapse_object = target._cell.esyn
                elif self.synapse_type == "inhibitory":
                    synapse_object = target._cell.isyn
                else:
                    synapse_object = getattr(target._cell, self.synapse_type)
                ###############source._cell.source.connect('event', synapse_object, 'synapse')
                synapse_object.n_incoming_connections += 1
                index = synapse_object.n_incoming_connections - 1
                synapse_object.setWeight(index, weight)
                synapse_object.setDelay(index, delay)'''
                index=0
                self.connections.append((source, target, index, weights, delays))

    def set(self, name, value):
        """
        Set connection attributes for all connections in this manager.

        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        #TODO: allow this!!
        #for conn in self.connections:
            #???


# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

set = common.set

initialize = common.initialize

####record = common.build_record('spikes', simulator)

####record_v = common.build_record('v', simulator)

####record_gsyn = common.build_record('gsyn', simulator)



def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    logger.debug("Being asked to record spikes of %s to %s"%(source, filename))

def record_v(source, filename):
    """Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    logger.debug("Being asked to record v of %s to %s"%(source, filename))

    global simNode, displayNode, color

    scale = "1"
    colour = colours[displayNode.childNodes.length%len(colours)]
    for i in range(source.size):
        lineNode = build_node('Line',
                              id=line_prefix+source.label,
                              scale=scale,
                              color=colour,
                              quantity="%s[%i]/%s"%(source.label,i,'v'),
                              save="%s_%i_%s_nml2.dat"%(source.label,i,'v'))

        displayNode.appendChild(lineNode)

def record_gsyn(source, filename):
    """Record gsyn."""
    print "Being asked to record gsyn of %s to %s"%(source, filename)

# ==============================================================================

## to reimplement in simulator.py...


min_delay = 0.0
max_delay = 1e12


def get_min_delay():
    """Return the minimum allowed synaptic delay."""
    return min_delay

def get_max_delay():
    """Return the maximum allowed synaptic delay."""
    return max_delay

common.get_min_delay = get_min_delay
common.get_max_delay = get_max_delay