# encoding: utf-8
"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
import nineml.user_layer as nineml
from pyNN import common, standardmodels, random, recording
from cells import *
from connectors import FixedProbabilityConnector, DistanceDependentProbabilityConnector
from utility import build_parameter_set, infer_units, catalog_url
import numpy


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
        self.assemblies = []

    def to_xml(self):
        return self.to_nineml().to_xml()

    def to_nineml(self):
        model = nineml.Model(name=self.label)
        for cs in self.current_sources:
            model.add_component(cs.to_nineml())
            # needToDefineWhichCellsTheCurrentIsInjectedInto
            # doWeJustReuseThePopulationProjectionIdiom="?"
        main_group = nineml.Group(name="Network")
        _populations = self.populations[:]
        _projections = self.projections[:]
        for a in self.assemblies:
            group = a.to_nineml()
            for p in a.populations:
                _populations.remove(p)
                group.add(p.to_nineml())
            for prj in self.projections:
                if (prj.pre is a or prj.pre in a.populations) and \
                   (prj.post is a or prj.post in a.populations):
                    _projections.remove(prj)
                    group.add(prj.to_nineml())
            model.add_group(group)
        for p in _populations:
            main_group.add(p.to_nineml())
        for prj in _projections:
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
    return 0
    
def end(compatible_output=True):
    """Write the XML file. Do any necessary cleaning up before exiting."""
    global net
    net.to_nineml().write(output_filename)
    

get_current_time, get_time_step, get_min_delay, get_max_delay, \
            num_processes, rank = common.build_state_queries(simulator)

def run(tstop):
    pass

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__
    
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def get_native_parameters(self):
        """Return a dictionary of parameters for the NEURON cell model."""
        return self._cell
    
    def set_native_parameters(self, parameters):
        """Set parameters of the NEURON cell model from a dictionary."""
        self._cell.update(parameters)


class Recorder(recording.Recorder):
    
    def _record(self, *args, **kwargs):
        pass
    
    def _local_count(self, filter):
        return 0


class BasePopulation(common.BasePopulation):

    def __getitem__(self, index):
        """
        Return a representation of the cell with the given index,
        suitable for being passed to other methods that require a cell id.
        Note that __getitem__ is called when using [] access, e.g.
            p = Population(...)
            p[2] is equivalent to p.__getitem__(2).
        Also accepts slices, e.g.
            p[3:6]
        which returns an array of cells.
        """
        if isinstance(index, int):
            return self.all_cells[index]
        elif isinstance(index, (slice, list, numpy.ndarray)):
            return PopulationView(self, index)
        elif isinstance(index, tuple):
            return PopulationView(self, list(index))
        else:
            raise Exception()

    def _record(self, variable, record_from, rng, to_file):
        pass

    def mean_spike_count(self, gather=True):
        return 0

    def printSpikes(self, file, gather=True, compatible_output=True):
        pass

    def print_v(self, file, gather=True, compatible_output=True):
        pass

    def rset(self, parametername, rand_distr):
        assert isinstance(rand_distr, random.RandomDistribution)
        translated_name = self.celltype.translations[parametername]['translated_name']
        self.celltype.parameters[translated_name] = rand_distr

    def initialize(self, variable, value):
        pass
    
    def get_synaptic_response_components(self, synaptic_mechanism_name):
        return [self.celltype.synapse_type_to_nineml(synaptic_mechanism_name, self.label)]
        

class Population(BasePopulation, common.Population):
    recorder_class = Recorder
    
    def __init__(self, size, cellclass, cellparams=None, structure=None, label=None):
        global net
        common.Population.__init__(self, size, cellclass, cellparams, structure, label) 
        net.populations.append(self)
    
    def _create_cells(self, cellclass, cellparams, size):
        celltype = cellclass(cellparams)
        self.all_cells = numpy.array([ID(i) for i in range(size)], dtype=ID)
        self._mask_local = numpy.ones(size, dtype=bool)
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        for id in self.all_cells:
            id.parent = self
            id._cell = celltype.parameters.copy()

    def to_nineml(self):
        if self.structure:
            structure = nineml.Structure(
                                    name="structure for %s" % self.label,
                                    definition=nineml.Definition("%s/networkstructures/%s.xml" % (catalog_url, self.structure.__class__.__name__)),
                                    parameters=build_parameter_set(self.structure.get_parameters())
                                    )
        else:
            structure = None
        population = nineml.Population(
                                name=self.label,
                                number=len(self),
                                prototype=self.celltype.to_nineml(self.label)[0],
                                positions=nineml.PositionList(structure=structure))
        return population


class PopulationView(BasePopulation, common.PopulationView):
    
    def __init__(self, parent, selector, label=None):
        global net
        common.PopulationView.__init__(self, parent, selector, label)
        net.populations.append(self)
    
    def to_nineml(self):
        selection = nineml.Selection(self.label,
                        nineml.All(
                            nineml.Eq("population[@name]", self.parent.label),
                            nineml.In("population[@id]", "%s:%s:%s" % (self.mask.start or "", self.mask.stop or "", self.mask.step or ""))
                        )
                    )
        return selection


class Assembly(common.Assembly):
    
    def __init__(self, label=None, *populations):
        global net
        common.Assembly.__init__(self, label, *populations)
        net.assemblies.append(self)
    
    def get_synaptic_response_components(self, synaptic_mechanism_name):
        components = set([])
        for p in self.populations:
            components.add(p.celltype.synapse_type_to_nineml(synaptic_mechanism_name, self.label))
        return components

    def to_nineml(self):
        group = nineml.Group(self.label)
        for p in self.populations:
            group.add(p.to_nineml())
        return group


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
                self.label = "%s---%s" % (self.pre.label, self.post.label)
        net.projections.append(self)
    
    def __len__(self):
        return 0
    
    def size(self):
        return len(self)
    
    def saveConnections(self, filename, gather=True, compatible_output=True):
        f = open(filename, 'w')
        f.write("At present, the 9ML backend is not able to save connections.")
        f.close()
    
    def to_nineml(self):
        connection_rule = self._method.to_nineml(self.label)
        connection_type = nineml.ConnectionType(
                                    name="connection type for projection %s" % self.label,
                                    definition=nineml.Definition("%s/connectiontypes/static_synapse.xml" % catalog_url),
                                    parameters=build_parameter_set(
                                                 {"weight": self._method.weights,
                                                  "delay": self._method.delays}))
        synaptic_responses = self.post.get_synaptic_response_components(self.target)
        synaptic_response, = synaptic_responses
        projection = nineml.Projection(
                                name=self.label,
                                source=self.pre.to_nineml(), # or just pass ref, and then resolve later?
                                target=self.post.to_nineml(),
                                rule=connection_rule,
                                synaptic_response=synaptic_response,
                                connection_type=connection_type)
        return projection


   
    
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
