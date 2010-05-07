# encoding: utf-8
import nineml.user_layer as nineml
from pyNN import common, standardmodels, cells, connectors, random
from cells import *
from utility import build_parameter_set, infer_units, catalog_url





def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, standardmodels.StandardCellType)]

   

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
        return self.to_nineml().to_xml()

    def to_nineml(self):
        model = nineml.Model(name=self.label)
        for cs in self.current_sources:
            model.add_component(cs.to_nineml())
            # needToDefineWhichCellsTheCurrentIsInjectedInto
            # doWeJustReuseThePopulationProjectionIdiom="?"
        main_group = nineml.Group(name="Network")
        for p in self.populations:
            main_group.add(p.to_nineml())
        for prj in self.projections:
            main_group.add(prj.to_nineml())
        model.add_group(main_group)
        return model


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
    net.to_nineml().write(output_filename)
    
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

    def to_nineml(self):
        structure = nineml.Structure(
                                name="grid for %s" % self.label,
                                definition=nineml.Definition("%s/networkstructures/%dDgrid.xml" % (catalog_url, self.ndim)),
                                parameters=build_parameter_set(get_grid_parameters(self)))
        population = nineml.Population(
                                name=self.label,
                                number=len(self),
                                prototype=self.celltype.to_nineml(self.label)[0],
                                positions=nineml.PositionList(structure=structure))
        return population


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
    
    def to_nineml(self):
        connection_rule = self._method.to_nineml(self.label)
        connection_type = nineml.ConnectionType(
                                    name="connection type for projection %s" % self.label,
                                    definition=nineml.Definition("%s/connectiontypes/static_synapse.xml" % catalog_url),
                                    parameters=build_parameter_set(
                                                 {"weight": self._method.weights,
                                                  "delay": self._method.delays}))
        synaptic_response = [c for c in self.post.celltype.to_nineml(self.post.label) if self.target in c.name][0] # this is a fragile hack
        projection = nineml.Projection(
                                name=self.label,
                                source=self.pre.label,
                                target=self.post.label,
                                rule=connection_rule,
                                synaptic_response=synaptic_response,
                                connection_type=connection_type)
        return projection

   
class ConnectorMixin(object):
    
    def to_nineml(self, label):
        connector_parameters = {}
        for name in self.__class__.parameter_names:
            connector_parameters[name] = getattr(self, name)
        connection_rule = nineml.ConnectionRule(
                                    name="connection rule for projection %s" % label,
                                    definition=nineml.Definition(self.definition_url),
                                    parameters=build_parameter_set(connector_parameters))
        return connection_rule


class FixedProbabilityConnector(connectors.FixedProbabilityConnector, ConnectorMixin):
    definition_url = "%s/connectionrules/fixed_probability.xml" % catalog_url 
    parameter_names = ('p_connect', 'allow_self_connections')
   
   
class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector, ConnectorMixin):
    definition_url = "%s/connectionrules/distance_dependent_probability.xml" % catalog_url
    parameter_names = ('d_expression', 'allow_self_connections') # space
   
    
class CurrentSource(object):
    """Base class for a source of current to be injected into a neuron."""
    counter = 0
    
    def __init__(self):
        global net
        net.current_sources.append(self)
        self.__class__.counter += 1

    def to_nineml(self):
        return nineml.CurrentSourceType(
                            name="current source %d" % self.__class__.counter,
                            definition=nineml.Definition("%s/currentsources/%s" % (catalog_url, self.definition_file)),
                            parameters=build_parameter_set(self.parameters))


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
