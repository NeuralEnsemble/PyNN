# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

$Id$
"""
import nest
from pyNN.nest2 import simulator
from pyNN import common, recording, __doc__
common.simulator = simulator
from pyNN.random import *
import numpy, os, shutil, logging, tempfile
from pyNN.nest2.cells import *
from pyNN.nest2.connectors import *
from pyNN.nest2.synapses import *
from pyNN.nest2.electrodes import *

Set = set


tempdirs       = []

NEST_SYNAPSE_TYPES = ["cont_delay_synapse" ,"static_synapse", "stdp_pl_synapse_hom",
                      "stdp_synapse", "stdp_synapse_hom", "tsodyks_synapse"]

# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]
    for cell_class in standard_cell_types:
        try:
            create(cell_class)
        except Exception, e:
            print "Warning: %s is defined, but produces the following error: %s" % (cell_class.__name__, e)
            standard_cell_types.remove(cell_class)
    return standard_cell_types

def _discrepancy_due_to_rounding(parameters, output_values):
    """NEST rounds delays to the time step."""
    if 'delay' not in parameters:
        return False
    else:
        # the logic here is not the clearest, the aim was to keep
        # _set_connection() as simple as possible, but it might be better to
        # refactor the whole thing.
        input_delay = parameters['delay']
        if hasattr(output_values, "__len__"):
            output_delay = output_values[parameters.keys().index('delay')]
        else:
            output_delay = output_values
        return abs(input_delay - output_delay) < get_time_step()

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global tempdir
    
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    
    if 'verbosity' in extra_params:
        nest_verbosity = extra_params['verbosity'].upper()
    else:
        nest_verbosity = "WARNING"
    nest.sli_run("M_%s setverbosity" % nest_verbosity)
        
    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')
    
    # reset the simulation kernel
    nest.ResetKernel()
    
    # set tempdir
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list
    nest.SetKernelStatus({'data_path': tempdir,})

    # set kernel RNG seeds
    num_threads = extra_params.get('threads') or 1
    if 'rng_seeds' in extra_params:
        rng_seeds = extra_params['rng_seeds']
    else:
        rng_seeds_seed = extra_params.get('rng_seeds_seed') or 42
        rng = NumpyRNG(rng_seeds_seed)
        rng_seeds = (rng.rng.uniform(size=num_threads*num_processes())*100000).astype('int').tolist() 
    logging.debug("rng_seeds = %s" % rng_seeds)
    nest.SetKernelStatus({'local_num_threads': num_threads,
                          'rng_seeds'        : rng_seeds})

    # Set min_delay and max_delay for all synapse models
    for synapse_model in NEST_SYNAPSE_TYPES:
        nest.SetDefaults(synapse_model, {'delay' : min_delay,
                                         'min_delay': min_delay,
                                         'max_delay': max_delay})

    # set resolution
    nest.SetKernelStatus({'resolution': timestep})
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global tempdirs
    # And we postprocess the low level files opened by record()
    # and record_v() method
    for recorder in simulator.recorder_list:
        recorder.write(gather=False, compatible_output=compatible_output)
    for tempdir in tempdirs:
        shutil.rmtree(tempdir)
    tempdirs = []

def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.run(simtime)
    return get_current_time()

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time = common.get_current_time
get_time_step = common.get_time_step
get_min_delay = common.get_min_delay
get_max_delay = common.get_max_delay
num_processes = common.num_processes
rank = common.rank

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.create

connect = common.connect

set = common.set

record = common.build_record('spikes', simulator)

record_v = common.build_record('v', simulator)

record_gsyn = common.build_record('gsyn', simulator)

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

    def __init__(self, dims, cellclass, cellparams=None, label=None):
        """
        Create a population of neurons all of the same type.
        
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
        common.Population.__init__(self, dims, cellclass, cellparams, label)

        # Should perhaps use "LayoutNetwork"?
        if isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
            self.celltype = cellclass(cellparams)
        
        self.all_cells, self._mask_local, self.first_id, self.last_id = simulator.create_cells(cellclass, cellparams, self.size, parent=self)
        self.local_cells = self.all_cells[self._mask_local]
        self.all_cells = self.all_cells.reshape(self.dim)
        self._mask_local = self._mask_local.reshape(self.dim)
        
        for id in self.local_cells:
            id.parent = self
        self.cell = self.all_cells # temporary alias, awaiting harmonization
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.recorders = {}
        for variable in simulator.RECORDING_DEVICE_NAMES:
            self.recorders[variable] = simulator.Recorder(variable, population=self)
        Population.nPop += 1

    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population.
        
        param can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param, str):
            if isinstance(val, (str, float, int)):
                param_dict = {param: float(val)}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param,dict):
            param_dict = param
        else:
            raise common.InvalidParameterValueError
        
        # The default implementation in common is is not very efficient for
        # simple and scaled parameters.
        # Should call nest.SetStatus(self.local_cells,...) for the parameters in
        # self.celltype.__class__.simple_parameters() and .scaled_parameters()
        # and keep the loop below just for the computed parameters. Even in this
        # case, it may be quicker to test whether the parameters participating
        # in the computation vary between cells, since if this is not the case
        # we can do the computation here and use nest.SetStatus.
        for key, value in param_dict.items():
            if not isinstance(self.celltype, str):
                # Here we check the consistency of the given parameters
                try:
                    self.celltype.default_parameters[key]
                except Exception:
                    raise common.NonExistentParameterError(key, self.celltype.__class__)
                if type(value) != type(self.celltype.default_parameters[key]):
                    raise common.InvalidParameterValueError
                
                # Then we do the call to SetStatus
                if key == 'v_init':
                    for cell in self.local_cells:
                        cell._v_init = value
                    nest.SetStatus(self.local_cells, "V_m", val) # not correct, since could set v_init in the middle of a simulation
                elif key in self.celltype.scaled_parameters():
                    translation = self.celltype.translations[key]
                    value = eval(translation['forward_transform'], globals(), {key:value})
                    nest.SetStatus(self.local_cells,translation['translated_name'],value)
                elif key in self.celltype.simple_parameters():
                    translation = self.celltype.translations[key]
                    nest.SetStatus(self.local_cells, translation['translated_name'], value)
                else:
                    for cell in self.local_cells:
                        cell.set_parameters(**{key:value})
            else:
                try:
                    nest.SetStatus(self.local_cells, key, value)
                except Exception:
                    raise common.InvalidParameterValueError

    #def rset(self, parametername, rand_distr):
    #    """
    #    'Random' set. Set the value of parametername to a value taken from
    #    rand_distr, which should be a RandomDistribution object.
    #    """
    #    if isinstance(rand_distr.rng, NativeRNG):
    #        raise Exception('rset() not yet implemented for NativeRNG')
    #    else:
    #        #rarr = rand_distr.next(n=len(self.local_cells))
    #        rarr = rand_distr.next(n=self.size)
    #        assert len(rarr) >= len(self.local_cells), "The length of rarr (%d) must be greater than that of local_cells (%d)" % (len(rarr), len(self.local_cells))
    #        rarr = rarr[:len(self.local_cells)]
    #        if not isinstance(self.celltype, str):
    #            try:
    #                self.celltype.default_parameters[parametername]
    #            except Exception:
    #                raise common.NonExistentParameterError(parametername, self.celltype.__class__)
    #            if parametername == 'v_init':
    #                for cell,val in zip(self.local_cells, rarr):
    #                    cell._v_init = val
    #                nest.SetStatus(self.local_cells, "V_m", rarr) # not correct, since could set v_init in the middle of a simulation
    #            elif parametername in self.celltype.scaled_parameters():
    #                translation = self.celltype.translations[parametername]
    #                rarr = eval(translation['forward_transform'], globals(), {parametername : rarr})
    #                nest.SetStatus(self.local_cells,translation['translated_name'],rarr)
    #            elif parametername in self.celltype.simple_parameters():
    #                translation = self.celltype.translations[parametername]
    #                nest.SetStatus(self.local_cells, translation['translated_name'], rarr)
    #            else:
    #                for cell,val in zip(self.local_cells, rarr):
    #                    setattr(cell, parametername, val)
    #        else:
    #           nest.SetStatus(self.local_cells, parametername, rarr)

    def _record(self, variable, record_from=None, rng=None, to_file=True):
        common.Population._record(self, variable, record_from, rng, to_file)
        nest.SetStatus(self.recorders[variable]._device, {'to_file': to_file, 'to_memory' : not to_file})
    
    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # Routine to give an average firing rate over all the threads/nodes
        # This is a rough approximation, because in fact each nodes is only multiplying 
        # the frequency of the recorders by the number of processes. To do better, we need a MPI
        # package to send informations to node 0. Nevertheless, it works for threaded mode
        if gather:
            node_list = range(nest.GetKernelStatus()["total_num_virtual_procs"])
        else:
            node_list = [rank()]
        n_spikes  = 0
        for node in node_list:
            nest.sps(self.recorders['spikes']._device[0])
            nest.sr("%d GetAddress %d append" %(self.recorders['spikes']._device[0], node))
            nest.sr("GetStatus /n_events get")
            n_spikes += nest.spp()
        n_rec = len(self.recorders['spikes'].recorded)
        if gather and num_processes()>1:
            n_rec = recording.mpi_sum(n_rec)
        return float(n_spikes)/n_rec

    def getSubPopulation(self, cell_list, label=None):
        
        # We get the dimensions of the new population
        dims = numpy.array(cell_list).shape
        # We create an empty population
        pop = Population(dims, cellclass=self.celltype, label=label, parent=self)
        # And then copy parameters from its parent
        pop.cellparams  = pop.parent.cellparams
        pop.first_id    = pop.parent.first_id
        idx             = numpy.array(cell_list,int).flatten() - pop.first_id
        pop.cell        = pop.parent.cell.flatten()[idx].reshape(dims)
        pop.local_cells  = pop.parent.local_cells[idx]
        pop.positions   = pop.parent.positions[:,idx]
        return pop


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """

    nProj = 0

    def __init__(self, presynaptic_population, postsynaptic_population,
                 method, source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.

        source - string specifying which attribute of the presynaptic cell
                 signals action potentials

        target - string specifying which synapse on the postsynaptic cell to
                 connect to

        If source and/or target are not given, default values are used.

        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.

        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.

        rng - specify an RNG object to be used by the Connector.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)

        self.synapse_type = target or 'excitatory'

        if isinstance(self.long_term_plasticity_mechanism, Set):
            logging.warning("Several STDP models are available for these connections:")
            logging.warning(", ".join(model for model in self.long_term_plasticity_mechanism))
            self.long_term_plasticity_mechanism = list(self.long_term_plasticity_mechanism)[0]
            logging.warning("By default, %s is used" % self.long_term_plasticity_mechanism)

        if synapse_dynamics and synapse_dynamics.fast and synapse_dynamics.slow:
                raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
        self._plasticity_model = self.short_term_plasticity_mechanism or \
                                 self.long_term_plasticity_mechanism or \
                                 "static_synapse"
        assert self._plasticity_model in NEST_SYNAPSE_TYPES, self._plasticity_model

        # Set synaptic plasticity parameters 
        # We create a particular synapse context just for this projection, by copying
        # the one which is desired. The name of the synapse context is randomly generated
        # and will be available as projection.plasticity_name
        self.plasticity_name = "projection_%d" % Projection.nProj
        Projection.nProj += 1
        synapse_defaults = nest.GetDefaults(self._plasticity_model)
        synapse_defaults.pop('synapsemodel')
        synapse_defaults.pop('num_connections')
        if 'num_connectors' in synapse_defaults:
            synapse_defaults.pop('num_connectors')
            
        if hasattr(self, '_short_term_plasticity_parameters') and self._short_term_plasticity_parameters:
            synapse_defaults.update(self._short_term_plasticity_parameters)

        if hasattr(self, '_stdp_parameters') and self._stdp_parameters:
            # NEST does not support w_min != 0
            self._stdp_parameters.pop("w_min_always_zero_in_NEST")
            # Tau_minus is a parameter of the post-synaptic cell, not of the connection
            tau_minus = self._stdp_parameters.pop("tau_minus")
            # The following is a temporary workaround until the NEST guys stop renaming parameters!
            if 'tau_minus' in nest.GetStatus([self.post.local_cells[0]])[0]:
                nest.SetStatus(self.post.local_cells, [{'tau_minus': tau_minus}])
            elif 'Tau_minus' in nest.GetStatus([self.post.local_cells[0]])[0]:
                nest.SetStatus(self.post.local_cells, [{'Tau_minus': tau_minus}])
            else:
                raise Exception("Postsynaptic cell model does not support STDP.")

            synapse_defaults.update(self._stdp_parameters)

        nest.CopyModel(self._plasticity_model, self.plasticity_name, synapse_defaults)
        self.connection_manager = simulator.ConnectionManager(self.plasticity_name, parent=self)
        
        # Create connections
        method.connect(self)

        self.connections = self.connection_manager


    # --- Methods for writing/reading information to/from file. ----------------

    def _dump_connections(self):
        """For debugging."""
        print "Connections for Projection %s, connected with %s" % (self.label or '(un-labelled)',
                                                                    self._method)
        print "\tsource\ttarget\tport"
        for src,tgt in zip(self._sources, self._targets):
            connections = nest.FindConnections([src],[tgt],self.plasticity_name)
            for port in connections['ports']:
                print "\t%d\t%d\t%d" % (src, tgt, port)
        print "Connection data for the presynaptic population (%s)" % self.pre.label
        for src in self.pre.cell.flat:
            print src, nest.GetConnections([src], self.plasticity_name)  



# ==============================================================================
