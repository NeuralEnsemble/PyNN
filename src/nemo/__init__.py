# -*- coding: utf-8 -*-
"""
Nemo implementation of the PyNN API.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: __init__.py 927 2011-02-03 16:56:10Z pierre $
"""

import logging
Set = set
import nemo
from pyNN.nemo import simulator
from pyNN import common, recording, space, core, __doc__
from pyNN.random import *
from pyNN.recording import files
from pyNN.nemo.standardmodels.cells import *
from pyNN.nemo.connectors import *
from pyNN.nemo.standardmodels.synapses import *
from pyNN.nemo.standardmodels.electrodes import *
from pyNN.nemo import electrodes
from pyNN.nemo.recording import *
from pyNN import standardmodels

logger = logging.getLogger("PyNN")

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, standardmodels.StandardCellType)]
    for cell_class in standard_cell_types:
        try:
            create(cell_class)
        except Exception, e:
            print "Warning: %s is defined, but produces the following error: %s" % (cell_class.__name__, e)
            standard_cell_types.remove(cell_class)
    return [obj.__name__ for obj in standard_cell_types]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=1, min_delay=1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    if (timestep < 1):
        raise Exception("It is not currently possible to have a timestep less than 1ms with this simulator")
    if (min_delay < 1):
        raise Exception("It is not currently possible to have a min_delay less than 1ms with this simulator")
    
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state = simulator._State(timestep, min_delay, max_delay)
    simulator.spikes_array_list = []
    simulator.recorder_list     = []
    if "cpu_backend" in extra_params:
	simulator.state.conf.set_cpu_backend()
    if "cuda_backend" in extra_params:
	simulator.state.conf.set_cuda_backend(extra_params["cuda_backend"])
    print "The backend used by nemo is: ", simulator.state.conf.backend_description()
    return simulator.state.mpi_rank

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    del simulator.state
    simulator.spikes_array_list = []
    simulator.recorder_list    = []
    electrodes.current_sources = []

    
def run(simtime):    
    """Run the simulation for simtime ms."""
    simulator.state.run(simtime)
    return simulator.state.t

reset      = simulator.reset
initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time, get_time_step, get_min_delay, get_max_delay, \
            num_processes, rank = common.build_state_queries(simulator)

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _simulator = simulator
    assembly_class = Assembly
    
    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population, common.BasePopulation):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    _simulator = simulator
    recorder_class = Recorder
    assembly_class = Assembly

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _create_cells(self, cellclass, cellparams=None, n=1):
        assert n > 0, 'n must be a positive integer'
        celltype = cellclass(cellparams)
        params   = celltype.parameters        
        self.all_cells  = numpy.array([simulator.ID(simulator.state.next_id) for cell in xrange(n)], simulator.ID)
        for cell in self.all_cells:
            cell.parent = self
        
        ntype = simulator.state.net.add_neuron_type(cellclass.nemo_name)
        i1    = numpy.ones(n)
        i0    = numpy.zeros(n).tolist()        
        init  = celltype.default_initial_values

        if isinstance(celltype, SpikeSourcePoisson):    
            simulator.state.net.add_neuron(ntype, self.all_cells.tolist(), params['rate']*i1)
        elif isinstance(celltype, SpikeSourceArray):
            ### For the moment, we model spike_source_array as neurons that are forced to fire, 
            ### but this could be enhanced. A local copy of these devices is kept on the CPU, to send the
            ### spikes
            simulator.spikes_array_list += self.all_cells.tolist()
            params['precision'] = simulator.state.dt
            for idx in self.all_cells:
                player = SpikeSourceArray.spike_player(**params)
                setattr(idx, 'player', player)                
            simulator.state.net.add_neuron(ntype, self.all_cells.tolist())
        elif isinstance(celltype, cells.IF_curr_exp) or isinstance(celltype, cells.IF_curr_alpha):            
            simulator.state.net.add_neuron(ntype, self.all_cells.tolist(),
                        list(params['v_rest']*i1),
                        list(params['v_reset']*i1),
                        list(params['cm']*i1),
                        list(params['tau_m']*i1),                    
                        list(params['t_refrac']*i1),
                        list(params['tau_syn_E']*i1),
                        list(params['tau_syn_I']*i1),
                        list(params['v_thresh']*i1),
                        list(params['i_offset']*i1),                        
                        list(init['v']*i1), i0, i0, list(1000.*i1))
        elif isinstance(celltype, cells.IF_cond_alpha):
            simulator.state.net.add_neuron(ntype, self.all_cells.tolist(),
                        list(params['v_rest']*i1),
                        list(params['v_reset']*i1),
                        list(params['cm']*i1),
                        list(params['tau_m']*i1),                    
                        list(params['t_refrac']*i1),
                        list(params['tau_syn_E']*i1),
                        list(params['tau_syn_I']*i1),
                        list(params['v_thresh']*i1),
                        list(params['i_offset']*i1),  
                        list(params['e_rev_E']*i1),  
                        list(params['e_rev_I']*i1),                      
                        list(init['v']*i1), i0, i0, i0, i0, list(1000.*i1))
        elif isinstance(celltype, cells.IF_cond_exp):
            simulator.state.net.add_neuron(ntype, self.all_cells.tolist(),
                        list(params['v_rest']*i1),
                        list(params['v_reset']*i1),
                        list(params['cm']*i1),
                        list(params['tau_m']*i1),                    
                        list(params['t_refrac']*i1),
                        list(params['tau_syn_E']*i1),
                        list(params['tau_syn_I']*i1),
                        list(params['v_thresh']*i1),
                        list(params['i_offset']*i1),  
                        list(params['e_rev_E']*i1),  
                        list(params['e_rev_I']*i1),                      
                        list(init['v']*i1), i0, i0, list(1000.*i1))
        elif isinstance(celltype, cells.Izikevich):            
            simulator.state.net.add_neuron(ntype, self.all_cells.tolist(), 
                        list(params['a']*i1), 
                        list(params['b']*i1), 
                        list(params['c']*i1), 
                        list(params['d']*i1), 
                        i0, 
                        list(init['u']*i1), 
                        list(init['v']*i1))
       
        self._mask_local = numpy.ones((n,), bool) # all cells are local
        self.first_id    = self.all_cells[0]
        self.last_id     = self.all_cells[-1]

    def _set_initial_value_array(self, variable, value):
        if not hasattr(value, "__len__"):
            value = value*numpy.ones((len(self),))
        indices = self.celltype.initial_indices
        if simulator.state.simulation is None:
            simulator.state.net.set_neuron_state(self.all_cells.tolist(), indices[variable], value.tolist()) 
        else:
            simulator.state.sim.set_neuron_state(self.all_cells.tolist(), indices[variable], value.tolist())


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    _simulator = simulator
    
    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentialss
                 
        target - string specifying which synapse on the postsynaptic cell to
                 connect to
                 
        If source and/or target are not given, default values are used.
        
        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.
        
        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.
        
        rng - specify an RNG object to be used by the Connector.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method,
                                   source, target, synapse_dynamics, label, rng)
        
        self._method           = method
        self._connections      = None
        self.synapse_type      = target or 'excitatory'
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                if self.synapse_dynamics.slow:
                    raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
                else:
                    raise Exception("Tsodyks Markram synapses not implemented in Nemo yet")
            elif self.synapse_dynamics.slow:
                self._plasticity_model = "stdp_synapse"
                if simulator.state.stdp is None:
                    simulator.state.set_stdp(self.synapse_dynamics.slow)
                if self.synapse_dynamics.slow == simulator.state.stdp:
                    pass                     
                else:
                    raise Exception("Only one STDP model can be handle by Nemo. Two detected !") 
        else:        
            self._plasticity_model = "static_synapse"
     
        self.synapse_model = self._plasticity_model
        self._sources      = []
        self._is_plastic   = False
        if self.synapse_model is "stdp_synapse":
            self._is_plastic = True
        self._connections = []
        
        method.connect(self)
        self._ranges = [0]
        for conn in self.connections:
            self._ranges += [self._ranges[-1] + (conn[1] - conn[0])]
        
    def __getitem__(self, i):
        idx = numpy.searchsorted(self._ranges, i)
        if isinstance(i, int):
            if i < len(self):
                padding = numpy.maximum(0, idx - 1)
                return simulator.Connection(self.connections[padding][0] + i - self._ranges[padding])
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                data = xrange(i.start, i.stop, i.step or 1)
                idx  = numpy.searchsorted(self._ranges, data)            
                res  = []
                pads = numpy.maximum(0, idx - 1)
                for count, j in enumerate(data):
                    res += [simulator.Connection(self.connections[pads[count]][0] + j - self._ranges[pads[count]])]
                return res
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))
    
    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return self._ranges[-1]

    @property
    def connections(self):        
        return self._connections

    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        #print "connecting", source, "to", targets, "with weights", weights, "and delays", delays
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        if not isinstance(source, common.IDMixin):
            raise errors.ConnectionError("source should be an ID object, actually %s" % type(source))
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        synapse_type = self.synapse_type or "excitatory"
        delays = numpy.array(delays).astype(numpy.int).tolist()
        if self.synapse_type == 'inhibitory' and common.is_conductance(targets[0]):
            weights *= -1 # NEMO wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()    
        source   = int(source)        
        synapses = simulator.state.net.add_synapse(source, targets, delays, weights, self._is_plastic)
        self._sources.append(source)
        self._connections += [(synapses[0], synapses[-1])]

    def get(self, parameter_name, format, gather=True):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this Projection.
        
        `parameter_name` -- name of the attribute whose values are wanted.
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.
        
        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if such a connection exists. If there are no such connections, X_ij will
        be NaN.
        """
        if parameter_name not in ('weight', 'delay'):
            raise Exception("Only weights and delays can be accessed by Nemo")

        if format == 'list':
            values = []
            for ranges in self.connections:
                if parameter_name is "weight":
                    values += simulator.state.sim.get_synapse_weight(xrange(ranges[0], ranges[1]))
                if parameter_name is "delay":
                    values += simulator.state.sim.get_synapse_delay(xrange(ranges[0], ranges[1]))
        elif format == 'array':
            value_arr = numpy.nan * numpy.ones((self.pre.size, self.post.size))            
            synapses  = []
            sources   = []
            targets   = []
            for ranges in self.connections:
                sources += simulator.state.sim.get_targets(xrange(ranges[0], ranges[1]))
                targets += simulator.state.sim.get_targets(xrange(ranges[0], ranges[1]))               
                if parameter_name is "weight":
                    data = simulator.state.sim.get_weights(xrange(ranges[0], ranges[1]))
                if parameter_name is "delay":
                    data = simulator.state.sim.get_delays(xrange(ranges[0], ranges[1]))          

            addr     = self.pre.id_to_index(sources), self.post.id_to_index(targets)   
            for idx in xrange(len(data)):
                address = addr[0][idx], addr[1][idx]
                if numpy.isnan(value_arr[address]):
                    value_arr[address] = data[idx]
                else:
                    value_arr[address] += data[idx]
            values = value_arr
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values

    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        lines = numpy.empty((len(self), 4))        
        synapses = []
        for count, ranges in enumerate(self.connections):
            idx          = range(self._ranges[count], self._ranges[count+1])
            lines[idx,0] = simulator.state.sim.get_sources(xrange(ranges[0], ranges[1]))
            lines[idx,1] = simulator.state.sim.get_targets(xrange(ranges[0], ranges[1]))  
            lines[idx,2] = simulator.state.sim.get_weights(xrange(ranges[0], ranges[1]))
            lines[idx,3] = simulator.state.sim.get_delays(xrange(ranges[0], ranges[1]))                   
        if compatible_output:
            lines[:,0] = self.pre.id_to_index(lines[:, 0]) 
            lines[:,1] = self.post.id_to_index(lines[:, 1])        
        file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
        file.close()                            

Space = space.Space

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

set = common.set

record      = common.build_record('spikes', simulator)
record_v    = common.build_record('v', simulator)
record_gsyn = common.build_record('gsyn', simulator)


# ==============================================================================
