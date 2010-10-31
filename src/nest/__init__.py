# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

$Id$
"""
import nest
from pyNN.nest import simulator
from pyNN import common, recording, errors, space, standardmodels, __doc__
common.simulator = simulator
recording.simulator = simulator

import numpy
import os
import shutil
import logging
import tempfile
from pyNN.nest.cells import *
from pyNN.nest.connectors import *
from pyNN.nest.synapses import *
from pyNN.nest.electrodes import *
from pyNN.nest.recording import *
from pyNN.random import RandomDistribution

Set = set
tempdirs       = []
#NEST_SYNAPSE_TYPES = ["cont_delay_synapse" ,"static_synapse", "stdp_pl_synapse_hom",
#                      "stdp_synapse", "stdp_synapse_hom", "tsodyks_synapse",
#                      "stdp_triplet_synapse", ]

NEST_SYNAPSE_TYPES = nest.Models(mtype='synapses')

STATE_VARIABLE_MAP = {"v": "V_m", "w": "w"}
logger = logging.getLogger("PyNN")

# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, standardmodels.StandardCellType)]
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

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global tempdir
    
    common.setup(timestep, min_delay, max_delay, **extra_params)
    
    if 'verbosity' in extra_params:
        nest_verbosity = extra_params['verbosity'].upper()
    else:
        nest_verbosity = "WARNING"
    nest.sli_run("M_%s setverbosity" % nest_verbosity)
    
    if "spike_precision" in extra_params:
        simulator.state.spike_precision = extra_params["spike_precision"]
    nest.SetKernelStatus({'off_grid_spiking': simulator.state.spike_precision=='off_grid'})
    
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
    logger.debug("rng_seeds = %s" % rng_seeds)
    nest.SetKernelStatus({'local_num_threads': num_threads,
                          'rng_seeds'        : rng_seeds})

    # set resolution
    nest.SetKernelStatus({'resolution': timestep})

    # Set min_delay and max_delay for all synapse models
    for synapse_model in NEST_SYNAPSE_TYPES:
        nest.SetDefaults(synapse_model, {'delay' : min_delay,
                                         'min_delay': min_delay,
                                         'max_delay': max_delay})
    simulator.connection_managers = []
    simulator.reset()
    
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global tempdirs
    # And we postprocess the low level files opened by record()
    # and record_v() method
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    for tempdir in tempdirs:
        shutil.rmtree(tempdir)
    tempdirs = []

def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.run(simtime)
    return get_current_time()

reset = common.reset

initialize = common.initialize

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
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    recorder_class = Recorder

    def _create_cells(self, cellclass, cellparams, n):
        """
        Create cells in NEST.
        
        `cellclass`  -- a PyNN standard cell or the name of a native NEST model.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        # this method should never be called more than once
        # perhaps should check for that
        assert n > 0, 'n must be a positive integer'
        if isinstance(cellclass, basestring):  # celltype is not a standard cell
            nest_model = cellclass
            cell_parameters = cellparams or {}
        elif isinstance(cellclass, type) and issubclass(cellclass, standardmodels.StandardCellType):
            celltype = cellclass(cellparams)
            nest_model = celltype.nest_name[simulator.state.spike_precision]
            cell_parameters = celltype.parameters
        else:
            raise Exception("Invalid cell type: %s" % type(cellclass))
        try:
            self.all_cells = nest.Create(nest_model, n)
        except nest.NESTError, err:
            if "UnknownModelName" in err.message and "cond" in err.message:
                raise errors.InvalidModelError("%s Have you compiled NEST with the GSL (Gnu Scientific Library)?" % err)
            raise errors.InvalidModelError(err)
        if cell_parameters:
            try:
                nest.SetStatus(self.all_cells, [cell_parameters])
            except nest.NESTError:
                print "NEST error when trying to set the following dictionary: %s" % cell_parameters
                raise
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        self._mask_local = numpy.array(nest.GetStatus(self.all_cells, 'local'))
        self.all_cells = numpy.array([simulator.ID(gid) for gid in self.all_cells], simulator.ID)
        for gid in self.all_cells:
            gid.parent = self

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
            elif isinstance(val, (list, numpy.ndarray)):
                param_dict = {param: [val]*len(self)}
            else:
                raise errors.InvalidParameterValueError
        elif isinstance(param,dict):
            param_dict = param
        else:
            raise errors.InvalidParameterValueError
        
        # The default implementation in common is is not very efficient for
        # simple and scaled parameters.
        # Should call nest.SetStatus(self.local_cells,...) for the parameters in
        # self.celltype.__class__.simple_parameters() and .scaled_parameters()
        # and keep the loop below just for the computed parameters. Even in this
        # case, it may be quicker to test whether the parameters participating
        # in the computation vary between cells, since if this is not the case
        # we can do the computation here and use nest.SetStatus.
        to_be_set = {}
        for key, value in param_dict.items():
            if not isinstance(self.celltype, str):
                # Here we check the consistency of the given parameters
                try:
                    self.celltype.default_parameters[key]
                except Exception:
                    raise errors.NonExistentParameterError(key, self.celltype.__class__)
                if type(value) != type(self.celltype.default_parameters[key]):
                    if isinstance(value, int) and isinstance(self.celltype.default_parameters[key], float):
                        value = float(value)
                    elif (isinstance(value, numpy.ndarray) and len(value.shape) == 1) and isinstance(self.celltype.default_parameters[key], list):
                        pass
                    else:
                        raise errors.InvalidParameterValueError("The parameter %s should be a %s, you supplied a %s" % (key,
                                                                                                                        type(self.celltype.default_parameters[key]),
                                                                                                                        type(value)))
                # Then we do the call to SetStatus
                if key in self.celltype.scaled_parameters():
                    translation = self.celltype.translations[key]
                    value = eval(translation['forward_transform'], globals(), {key:value})
                    to_be_set[translation['translated_name']] = value
                elif key in self.celltype.simple_parameters():
                    translation = self.celltype.translations[key]
                    to_be_set[translation['translated_name']] = value                    
                else:
                    to_be_set[key] = value
            else:
                try:
                    nest.SetStatus(self.local_cells, key, value)
                except Exception:
                    raise errors.InvalidParameterValueError
            nest.SetStatus(self.local_cells.tolist(), to_be_set)

    def initialize(self, variable, value):
        """
        Set the initial value of one of the state variables of the neurons in
        this population.
        
        `value` may either be a numeric value (all neurons set to the same
                value) or a `RandomDistribution` object (each neuron gets a
                different value)
        """
        if isinstance(value, RandomDistribution):
            rarr = value.next(n=self.all_cells.size, mask_local=self._mask_local)
            value = rarr #numpy.array(rarr)
            assert len(rarr) == len(self.local_cells), "%d != %d" % (len(rarr), len(self.local_cells))
        nest.SetStatus(self.local_cells.tolist(), STATE_VARIABLE_MAP[variable], value)
        self.initial_values[variable] = core.LazyArray(self.size, value)

    def _record(self, variable, record_from=None, rng=None, to_file=True):
        common.Population._record(self, variable, record_from, rng, to_file)
        # need to set output filename if supplied
        nest.SetStatus(self.recorders[variable]._device, {'to_file': bool(to_file), 'to_memory' : not to_file})


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
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                if self.synapse_dynamics.slow:
                    raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
                else:
                    self._plasticity_model = self.synapse_dynamics.fast.native_name
            elif synapse_dynamics.slow:
                self._plasticity_model = self.synapse_dynamics.slow.possible_models
                if isinstance(self._plasticity_model, Set):
                    logger.warning("Several STDP models are available for these connections:")
                    logger.warning(", ".join(model for model in self._plasticity_model))
                    self._plasticity_model = list(self._plasticity_model)[0]
                    logger.warning("By default, %s is used" % self._plasticity_model)
        else:        
            self._plasticity_model = "static_synapse"
        if self._plasticity_model not in NEST_SYNAPSE_TYPES:
            raise ValueError, "Synapse dynamics model '%s' not a valid NEST synapse model.  Possible models in your NEST build are: %s" % ( self._plasticity_model, str(nest.Models(mtype='synapses')))

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
            
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                synapse_defaults.update(self.synapse_dynamics.fast.parameters)
            elif self.synapse_dynamics.slow:
                stdp_parameters = self.synapse_dynamics.slow.all_parameters
                # NEST does not support w_min != 0
                stdp_parameters.pop("w_min_always_zero_in_NEST")
                # Tau_minus is a parameter of the post-synaptic cell, not of the connection
                tau_minus = stdp_parameters.pop("tau_minus")
                # The following is a temporary workaround until the NEST guys stop renaming parameters!
                if 'tau_minus' in nest.GetStatus([self.post.local_cells[0]])[0]:
                    nest.SetStatus(self.post.local_cells.tolist(), [{'tau_minus': tau_minus}])
                elif 'Tau_minus' in nest.GetStatus([self.post.local_cells[0]])[0]:
                    nest.SetStatus(self.post.local_cells.tolist(), [{'Tau_minus': tau_minus}])
                else:
                    raise Exception("Postsynaptic cell model does not support STDP.")

                synapse_defaults.update(stdp_parameters)

        nest.CopyModel(self._plasticity_model, self.plasticity_name, synapse_defaults)
        self.connection_manager = simulator.ConnectionManager(self.synapse_type,
                                                              self.plasticity_name, parent=self)
        
        # Create connections
        method.connect(self)

        self.connections = self.connection_manager

    def saveConnections(self, filename, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        import operator
        fmt = "%d\t%d\t%g\t%g\n"
        lines = []
        res   = nest.GetStatus(self.connection_manager.connections, ('source', 'target', 'weight', 'delay'))

        if not compatible_output:
            for c in res:   
                line = fmt  % (c[0], c[1], 0.001*c[2], c[3])
                lines.append(line)
        else:
            for c in res:   
                line = fmt  % (self.pre.id_to_index(c[0]), self.post.id_to_index(c[1]), 0.001*c[2], c[3])
                lines.append(line)
        if gather == True and num_processes() > 1:
            all_lines = { rank(): lines }
            all_lines = recording.gather_dict(all_lines)
            if rank() == 0:
                lines = reduce(operator.add, all_lines.values())
        elif num_processes() > 1:
            filename += '.%d' % rank()
        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)
        if gather == False or rank() == 0:
            f = open(filename, 'w')
            f.write(self.pre.label + "\n" + self.post.label + "\n")
            f.writelines(lines)
            f.close()

    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        self.setWeights(rand_distr.next(len(self), mask_local=False))

    def randomizeDelays(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        self.setDelays(rand_distr.next(len(self), mask_local=False))

Space = space.Space

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

set = common.set

record = common.build_record('spikes', simulator)

record_v = common.build_record('v', simulator)

record_gsyn = common.build_record('gsyn', simulator)

# ==============================================================================
