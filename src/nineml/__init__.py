# encoding: utf-8
from lxml.etree import ElementTree, Element, SubElement
from lxml.builder import E
from pyNN import common, standardmodels, cells, connectors, random

NNMDL_NAMESPACE = "http://nineml.org/namespace"
NNMDL = "{%s}" % NNMDL_NAMESPACE

units_map = {
    "time": "ms",
    "potential": "mV",
    "threshold": "mV",
    "capacitance": "nS",
    "frequency": "Hz",
    "duration": "ms",
    "onset": "ms",
    "amplitude": "nA", # dodgy. Probably better to include units with class definitions
    "weight": "dimensionless",
    "delay": "ms",
    "dx": u"µm", "dy": u"µm", "dz": u"µm",
    "x0": u"µm", "y0": u"µm", "z0": u"µm",
    "aspectratio": "dimensionless",
}

SPIKINGNODE = E.spikingNode
RANDOMDISTRIBUTION = E.randomDistribution
DEFINITION = lambda url: E.definition(E.url(url))
SYNAPSE = E.synapse
CURRENTSOURCE = E.currentSource
#def POPULATION(*args, **kwargs):
#    name = kwargs["name"]
#    return E.group(E.component(*args, **kwargs), name=name)
POPULATION = E.component # component is not a good name for this
PROJECTION = E.projection
REFERENCE = E.reference

def infer_units(parameter_name):
    unit = "unknown"
    for fragment, u in units_map.items():
        if fragment in parameter_name.lower():
            unit = u
            break
    return unit

def build_parameters_node(parameters, dimensionless=False): 
    parameters_node = Element("properties") # I personally think this element should be called "parameters", reserving "properties" for things like position, that can apply to any spiking node
    for name, value in parameters.items():
        if isinstance(value, random.RandomDistribution):
            rand_distr = value
            value_node = RANDOMDISTRIBUTION(
                            DEFINITION("http://nineml.org/library/%s_distribution.xml" % rand_distr.name),
                            build_parameters_node(map_random_distribution_parameters(rand_distr.name, rand_distr.parameters),
                                                  dimensionless=True),
                            name="%s(%s)" % (rand_distr.name, ",".join(str(p) for p in rand_distr.parameters))
                         )
        else:
            value_node = str(value)
        if dimensionless:
            unit = "dimensionless"
        elif isinstance(value, basestring):
            unit = None
        else:
            unit = infer_units(name)
        if unit is None:
            parameters_node.append(E(name,
                                     E.value(value_node)))
        else:
            parameters_node.append(E(name,
                                     E.value(value_node),
                                     E.unit(unit)))
    return parameters_node
    
def map_random_distribution_parameters(name, parameters):
    parameter_map = {
        'normal': ('mean', 'standardDeviation'),
        'uniform': ('lowerBound', 'upperBound'),
    }
    P = {}
    for name,val in zip(parameter_map[name], parameters):
        P[name] = val
    return P

def get_grid_parameters(population):
    P = {"fillOrder": "sequential"}
    dim_names = ["x", "y", "z"][:population.ndim]
    for n,label in zip(population.dim, dim_names):
        #P["n%s" % label] = n
        P["%s0" % label] = 0.0
        P["d%s" % label] = 1
    if population.ndim > 1:
        P["aspectRatioXY"] = population.dim[0]/population.dim[1]
        if population.ndim > 2:
            P["aspectRatioXZ"] = population.dim[0]/population.dim[2]
    return P


class Network(object):

    def __init__(self, label):
        self.label = label
        self.populations = []
        self.projections = []
        self.current_sources = []

    def to_xml(self):
        root = Element("nineml", xmlns=NNMDL_NAMESPACE, name=self.label)
        for p in self.populations:
            cell_parameters = build_parameters_node(p.celltype.spiking_mechanism_parameters)
            root.append(
                SPIKINGNODE(
                    E.definition(E.url("file://%s.xml" % p.celltype.__class__.__name__)),
                    cell_parameters,
                    name="%s_neuron_type" % p.label
                )
            )
        for p in self.populations:
            for synapse_type in p.celltype.synapse_types:
                root.append(
                    SYNAPSE(
                        DEFINITION("http://nineml.org/library/%s_syn.xml" % p.celltype.__class__.__name__),
                        build_parameters_node(p.celltype.synaptic_mechanism_parameters[synapse_type]),
                        name="%s %s synapse" % (p.label, synapse_type)
                    )
                )
        
        for cs in self.current_sources:
            root.append(
                CURRENTSOURCE(
                    DEFINITION("http://nineml.org/library/%s" % cs.definition_file),
                    build_parameters_node(cs.parameters),
                    name="current source (needs a unique identifier)"
                )
            )
            root.append(
                E.needToDefineWhichCellsTheCurrentIsInjectedInto(
                    doWeJustReuseThePopulationProjectionIdiom="?"
                )
            )
         
        main_group = SubElement(root, "group", name="Network")
        for p in self.populations:
            main_group.append(
                POPULATION(
                    E.number(str(len(p))),
                    REFERENCE("%s_neuron_type" % p.label),
                    E.positions(
                        E.structure(
                            DEFINITION("http://nineml.org/library/%dDgrid.xml" % p.ndim),
                            build_parameters_node(get_grid_parameters(p)),
                            name="grid for %s" % p.label
                        )
                    ),
                    name=p.label))
        
        for prj in self.projections:
            connector_parameters = []
            for name in prj._method.__class__.parameter_names:
                connector_parameters.append(E(name, E.value(str(getattr(prj._method, name)))))
            main_group.append(
                PROJECTION(
                    E.source(prj.pre.label),
                    E.target(prj.post.label),
                    E.rule(
                        DEFINITION("file://%s.xml" % prj._method.__class__.__name__),
                        *connector_parameters
                    ),
                    E.postSynapticResponse(
                        REFERENCE("%s %s synapse" % (prj.post.label, prj.target))
                    ),
                    E.connection(
                        DEFINITION("http://nineml.org/library/static_synapse.xml"),
                        build_parameters_node({"weight": prj._method.weights, "delay": prj._method.delays}),
                    ),
                    name=prj.label
                )
            )
        return root

class DummySimulator(object):
    class State(object):
        mpi_rank = 0
        min_delay = 1e99
        num_processes = 1
    state = State()
simulator = DummySimulator()
common.simulator = simulator

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.

    incf-specific extra_params:

    filename - output filename
    label - a name for the model

    returns: MPI rank

    """
    global output_filename, net, simulator
    simulator.state.min_delay = min_delay
    output_filename = extra_params["filename"]
    label = extra_params["label"]
    net = Network(label)
    
    
def end(compatible_output=True):
    """Write the XML file. Do any necessary cleaning up before exiting."""
    global net
    ElementTree(net.to_xml()).write(output_filename, encoding="UTF-8",
                                    pretty_print=True, xml_declaration=True)
    
    
get_min_delay = common.get_min_delay


class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__
    
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


class Population(common.Population):
    
    def __init__(self, dims, cellclass, cellparams=None, label=None):
        global net
        common.Population.__init__(self, dims, cellclass, cellparams, label)
        
        net.populations.append(self)
        
    def __getitem__(self, addr):
        """
        Return a representation of the cell with coordinates given by addr,
        suitable for being passed to other methods that require a cell id.
        Note that __getitem__ is called when using [] access, e.g.
            p = Population(...)
            p[2,3] is equivalent to p.__getitem__((2,3)).
        Also accepts slices, e.g.
            p[0,3:6]
        which returns an array of cells.
        """
        #if isinstance(addr, (int, slice)):
        #    addr = (addr,)
        #if len(addr) == self.ndim:
        id = addr # just for testing. Assume 1D population
        #else:
        #    raise errors.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim, str(addr))
        return ID(id)
    
    def rset(self, parametername, rand_distr):
        assert isinstance(rand_distr, random.RandomDistribution)
        translated_name = self.celltype.translations[parametername]['translated_name']
        self.celltype.parameters[translated_name] = rand_distr


class Projection(common.Projection):
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 method,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        global net
        common.Projection.__init__(self, presynaptic_population,
                                   postsynaptic_population, method, source,
                                   target, synapse_dynamics, label, rng)
        if label is None:
            if self.pre.label and self.post.label:
                self.label = "%s-%s" % (self.pre.label, self.post.label)
        net.projections.append(self)
        
        
class CellTypeMixin(object):
    
    @property
    def spiking_mechanism_parameters(self):
        smp = {}
        for name in self.__class__.spiking_mechanism_parameter_names:
            smp[name] = self.parameters[name]
        return smp
    
    @property
    def synaptic_mechanism_parameters(self):
        smp = {}
        for synapse_type in self.__class__.synapse_types:
            smp[synapse_type] = {}
            for name in self.__class__.synaptic_mechanism_parameter_names[synapse_type]:
                smp[synapse_type][name.split("_")[1]] = self.parameters[name]
        return smp
    
        
class IF_cond_exp(cells.IF_cond_exp, CellTypeMixin):
   
    translations = standardmodels.build_translations(
        ('tau_m',      'membraneTimeConstant'),
        ('cm',         'membraneCapacitance'),
        ('v_rest',     'restingPotential'),
        ('v_thresh',   'threshold'),
        ('v_reset',    'resetPotential'),
        ('tau_refrac', 'refractoryTime'),
        ('i_offset',   'offsetCurrent'),
        ('tau_syn_E',  'excitatory_decayTimeConstant'),
        ('tau_syn_I',  'inhibitory_decayTimeConstant'),
        ('v_init',     'initialMembranePotential'),
        ('e_rev_E',    'excitatory_reversalPotential'),
        ('e_rev_I',    'inhibitory_reversalPotential')
    )
    spiking_mechanism_parameter_names = ('membraneTimeConstant','membraneCapacitance',
                                         'restingPotential', 'threshold',
                                         'resetPotential', 'refractoryTime')
    synaptic_mechanism_parameter_names = {
        'excitatory': ['excitatory_decayTimeConstant', 'excitatory_reversalPotential'],
        'inhibitory': ['inhibitory_decayTimeConstant',  'inhibitory_reversalPotential']
    }
   

class FixedProbabilityConnector(connectors.FixedProbabilityConnector):
    parameter_names = ('p_connect', 'allow_self_connections')
   
   
class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector):
    parameter_names = ('d_expression', 'allow_self_connections') # space
   
class SpikeSourcePoisson(cells.SpikeSourcePoisson, CellTypeMixin):
    
    translations = standardmodels.build_translations(
        ('start',    'onset'),
        ('rate',     'frequency'),
        ('duration', 'duration'),
    )
    
    spiking_mechanism_parameter_names = ("onset", "frequency", "duration")

    
class CurrentSource(object):
    """Base class for a source of current to be injected into a neuron."""
    
    def __init__(self):
        global net
        net.current_sources.append(self)


class DCSource(CurrentSource):
    """Source producing a single pulse of current of constant amplitude."""
    definition_file = "current_pulse.xml"
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """Construct the current source.
        
        Arguments:
            start     -- onset time of pulse in ms
            stop      -- end of pulse in ms
            amplitude -- pulse amplitude in nA
        """
        CurrentSource.__init__(self)
        self.amplitude = amplitude
        self.start = start
        self.stop = stop or 1e12
        
    @property
    def parameters(self):
        return {"amplitude": self.amplitude,
                "onset": self.start,
                "duration": self.start+self.stop}
        
    
    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        self.cell_list = cell_list
