# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.


"""

import numpy
import nest
from pyNN.nest import simulator
from pyNN import common, recording, errors, space, __doc__
common.simulator = simulator
recording.simulator = simulator

if recording.MPI and (nest.Rank() != recording.mpi_comm.rank):
    raise Exception("MPI not working properly. Please make sure you import pyNN.nest before pyNN.random.")

import os
import shutil
import logging
import tempfile
from pyNN.recording import files
from pyNN.nest.cells import NativeCellType, native_cell_type
from pyNN.nest.synapses import NativeSynapseDynamics, NativeSynapseMechanism
from pyNN.nest.standardmodels.cells import *
from pyNN.nest.connectors import *
from pyNN.nest.standardmodels.synapses import *
from pyNN.nest.electrodes import *
from pyNN.nest.recording import *
from pyNN.random import RandomDistribution
from pyNN import standardmodels

Set = set
tempdirs       = []
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
    return [obj.__name__ for obj in standard_cell_types]

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
    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')
    
    # reset the simulation kernel
    nest.ResetKernel()
    
    if 'verbosity' in extra_params:
        nest_verbosity = extra_params['verbosity'].upper()
    else:
        nest_verbosity = "WARNING"
    nest.sli_run("M_%s setverbosity" % nest_verbosity)
    
    if "spike_precision" in extra_params:
        simulator.state.spike_precision = extra_params["spike_precision"]
        if extra_params["spike_precision"] == 'off_grid':
            simulator.state.default_recording_precision = 15
    nest.SetKernelStatus({'off_grid_spiking': simulator.state.spike_precision=='off_grid'})
    if "recording_precision" in extra_params:
        simulator.state.default_recording_precision = extra_params["recording_precision"]
        
    # all NEST to erase previously written files (defaut with all the other simulators)
    nest.SetKernelStatus({'overwrite_files' : True})
    
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

    if 'allow_offgrid_spikes' in nest.GetDefaults('spike_generator'):
        nest.SetDefaults('spike_generator', {'allow_offgrid_spikes': True})

    # Set min_delay and max_delay for all synapse models
    if min_delay != 'auto':
        NEST_SYNAPSE_TYPES = nest.Models(mtype='synapses')  # need to rebuild after ResetKernel
        for synapse_model in NEST_SYNAPSE_TYPES:
            nest.SetDefaults(synapse_model, {'delay' : min_delay,
                                             'min_delay': min_delay,
                                             'max_delay': max_delay})
    simulator.connection_managers = []
    simulator.populations = []
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
    simulator.recorder_list = []

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
        celltype = cellclass(cellparams)
        nest_model = celltype.nest_name[simulator.state.spike_precision]
        try:
            self.all_cells = nest.Create(nest_model, n, params=celltype.parameters)
        except nest.NESTError, err:
            if "UnknownModelName" in err.message and "cond" in err.message:
                raise errors.InvalidModelError("%s Have you compiled NEST with the GSL (Gnu Scientific Library)?" % err)
            raise errors.InvalidModelError(err)
        # create parrot neurons if necessary
        if hasattr(celltype, "uses_parrot") and celltype.uses_parrot:
            self.all_cells_source = numpy.array(self.all_cells)  # we put the parrots into all_cells, since this will
            self.all_cells = nest.Create("parrot_neuron", n)     # be used for connections and recording. all_cells_source
            nest.Connect(self.all_cells_source, self.all_cells)  # should be used for setting parameters
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        self._mask_local = numpy.array(nest.GetStatus(self.all_cells, 'local'))
        self.all_cells = numpy.array([simulator.ID(gid) for gid in self.all_cells], simulator.ID)
        for gid in self.all_cells:
            gid.parent = self
        if hasattr(celltype, "uses_parrot") and celltype.uses_parrot:
            for gid, source in zip(self.all_cells, self.all_cells_source):
                gid.source = source
        simulator.populations.append(self)
        

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
        elif isinstance(param, dict):
            param_dict = param
        else:
            raise errors.InvalidParameterValueError
        param_dict = self.celltype.checkParameters(param_dict, with_defaults=False)
        # The default implementation in common is is not very efficient for
        # simple and scaled parameters.
        # Should call nest.SetStatus(self.local_cells,...) for the parameters in
        # self.celltype.__class__.simple_parameters() and .scaled_parameters()
        # and keep the loop below just for the computed parameters. Even in this
        # case, it may be quicker to test whether the parameters participating
        # in the computation vary between cells, since if this is not the case
        # we can do the computation here and use nest.SetStatus.
        
        if isinstance(self.celltype, standardmodels.StandardCellType):
            to_be_set = {}
            if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
                gids = self.all_cells_source[self._mask_local]
            else:
                gids = self.local_cells
            for key, value in param_dict.items():
                if key in self.celltype.scaled_parameters():
                    translation = self.celltype.translations[key]
                    value = eval(translation['forward_transform'], globals(), {key:value})
                    to_be_set[translation['translated_name']] = value                
                elif key in self.celltype.simple_parameters():
                    translation = self.celltype.translations[key]
                    to_be_set[translation['translated_name']] = value                    
                else:
                    assert key in self.celltype.computed_parameters()
            logger.debug("Setting the following parameters: %s" % to_be_set)
            nest.SetStatus(gids.tolist(), to_be_set)
            for key, value in param_dict.items():
                if key in self.celltype.computed_parameters():
                    logger.debug("Setting %s = %s" % (key, value))
                    for cell in self:
                        cell.set_parameters(**{key:value})
        else:
            nest.SetStatus(self.local_cells.tolist(), param_dict)

    def _set_initial_value_array(self, variable, value):
        if variable in STATE_VARIABLE_MAP:
            variable = STATE_VARIABLE_MAP[variable]
        nest.SetStatus(self.local_cells.tolist(), variable, value)

PopulationView = common.PopulationView
Assembly = common.Assembly


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
            synapse_dynamics = self.synapse_dynamics
            self.synapse_dynamics._set_tau_minus(self.post.local_cells) 
        else:        
            synapse_dynamics = NativeSynapseDynamics("static_synapse")
        self.synapse_model = synapse_dynamics._get_nest_synapse_model("projection_%d" % Projection.nProj)
        Projection.nProj += 1
        self.connection_manager = simulator.ConnectionManager(self.synapse_type,
                                                              self.synapse_model,
                                                              parent=self)
        
        # Create connections
        method.connect(self)
        self.connection_manager._set_tsodyks_params()
        self.connections = self.connection_manager
        

    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        import operator
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        lines   = nest.GetStatus(self.connection_manager.connections, ('source', 'target', 'weight', 'delay'))  
        
        if gather == True and num_processes() > 1:
            all_lines = { rank(): lines }
            all_lines = recording.gather_dict(all_lines)
            if rank() == 0:
                lines = reduce(operator.add, all_lines.values())
        elif num_processes() > 1:
            file.rename('%s.%d' % (file.name, rank()))
        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)
                
        if gather == False or rank() == 0:
            lines       = numpy.array(lines, dtype=float)
            lines[:,2] *= 0.001
            if compatible_output:
                lines[:,0] = self.pre.id_to_index(lines[:,0])
                lines[:,1] = self.post.id_to_index(lines[:,1])  
            file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
            file.close()        

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
