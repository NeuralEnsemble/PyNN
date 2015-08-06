# encoding: utf-8
"""
Common implementation of ID, Population, PopulationView and Assembly classes.

These base classes should be sub-classed by the backend-specific classes.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import os
import logging
import operator
from itertools import chain
import tempfile
try:
    basestring
    reduce
except NameError:
    basestring = str
    from functools import reduce
from collections import defaultdict
from pyNN import random, recording, errors, standardmodels, core, space, descriptions
from pyNN.models import BaseCellType
from pyNN.parameters import ParameterSpace, LazyArray, simplify as simplify_parameter_array
from pyNN.recording import files

deprecated = core.deprecated
logger = logging.getLogger("PyNN")


def is_conductance(target_cell):
    """
    Returns True if the target cell uses conductance-based synapses, False if
    it uses current-based synapses, and None if the synapse-basis cannot be
    determined.
    """
    if hasattr(target_cell, 'local') and target_cell.local and hasattr(target_cell, 'celltype'):
        is_conductance = target_cell.celltype.conductance_based
    else:
        is_conductance = None
    return is_conductance


class IDMixin(object):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object.
    """
    # Simulator ID classes should inherit both from the base type of the ID
    # (e.g., int or long) and from IDMixin.

    def __getattr__(self, name):
        if name == "parent":
            raise Exception("parent is not set")
        try:
            val = self.get_parameters()[name]
        except KeyError:
            raise errors.NonExistentParameterError(name,
                                                   self.celltype.__class__.__name__,
                                                   self.celltype.get_parameter_names())
        return val

    def __setattr__(self, name, value):
        if name == "parent":
            object.__setattr__(self, name, value)
        elif self.celltype.has_parameter(name):
            self.set_parameters(**{name: value})
        else:
            object.__setattr__(self, name, value)

    def set_parameters(self, **parameters):
        """
        Set cell parameters, given as a sequence of parameter=value arguments.
        """
        # if some of the parameters are computed from the values of other
        # parameters, need to get and translate all parameters
        if self.local:
            self.as_view().set(**parameters)
        else:
            raise errors.NotLocalError("Cannot set parameters for a cell that does not exist on this node.")

    def get_parameters(self):
        """Return a dict of all cell parameters."""
        if self.local:
            parameter_names = self.celltype.get_parameter_names()
            return dict((k, v) for k,v in zip(parameter_names, self.as_view().get(parameter_names)))
        else:
            raise errors.NotLocalError("Cannot obtain parameters for a cell that does not exist on this node.")

    @property
    def celltype(self):
        return self.parent.celltype

    @property
    def is_standard_cell(self):
        return isinstance(self.celltype, standardmodels.StandardCellType)

    def _set_position(self, pos):
        """
        Set the cell position in 3D space.

        Cell positions are stored in an array in the parent Population.
        """
        assert isinstance(pos, (tuple, numpy.ndarray))
        assert len(pos) == 3
        self.parent._set_cell_position(self, pos)

    def _get_position(self):
        """
        Return the cell position in 3D space.

        Cell positions are stored in an array in the parent Population, if any,
        or within the ID object otherwise. Positions are generated the first
        time they are requested and then cached.
        """
        return self.parent._get_cell_position(self)

    position = property(_get_position, _set_position)

    @property
    def local(self):
        return self.parent.is_local(self)

    def inject(self, current_source):
        """Inject current from a current source object into the cell."""
        current_source.inject_into([self])

    def get_initial_value(self, variable):
        """Get the initial value of a state variable of the cell."""
        return self.parent._get_cell_initial_value(self, variable)

    def set_initial_value(self, variable, value):
        """Set the initial value of a state variable of the cell."""
        self.parent._set_cell_initial_value(self, variable, value)

    def as_view(self):
        """Return a PopulationView containing just this cell."""
        index = self.parent.id_to_index(self)
        return self.parent[index:index+1]


class BasePopulation(object):
    _record_filter = None

    def __getitem__(self, index):
        """
        Return either a single cell (ID object) from the Population, if `index`
        is an integer, or a subset of the cells (PopulationView object), if
        `index` is a slice or array.

        Note that __getitem__ is called when using [] access, e.g.
            p = Population(...)
            p[2] is equivalent to p.__getitem__(2).
            p[3:6] is equivalent to p.__getitem__(slice(3, 6))
        """
        if isinstance(index, (int, numpy.integer)):
            return self.all_cells[index]
        elif isinstance(index, (slice, list, numpy.ndarray)):
            return self._get_view(index)
        elif isinstance(index, tuple):
            return self._get_view(list(index))
        else:
            raise TypeError("indices must be integers, slices, lists, arrays or tuples, not %s" % type(index).__name__)

    def __len__(self):
        """Return the total number of cells in the population (all nodes)."""
        return self.size

    @property
    def local_size(self):
        """Return the number of cells in the population on the local MPI node"""
        return len(self.local_cells) # would self._mask_local.sum() be faster?

    def __iter__(self):
        """Iterator over cell ids on the local node."""
        return iter(self.local_cells)

    @property
    def conductance_based(self):
        """
        Indicates whether the post-synaptic response is modelled as a change
        in conductance or a change in current.
        """
        return self.celltype.conductance_based

    @property
    def receptor_types(self):
        return self.celltype.receptor_types

    def is_local(self, id):
        """
        Indicates whether the cell with the given ID exists on the local MPI node.
        """
        assert id.parent is self
        index = self.id_to_index(id)
        return self._mask_local[index]

    def all(self):
        """Iterator over cell ids on all MPI nodes."""
        return iter(self.all_cells)

    def __add__(self, other):
        """
        A Population/PopulationView can be added to another Population,
        PopulationView or Assembly, returning an Assembly.
        """
        assert isinstance(other, BasePopulation)
        return self._assembly_class(self, other)

    def _get_cell_position(self, id):
        index = self.id_to_index(id)
        return self.positions[:, index]

    def _set_cell_position(self, id, pos):
        index = self.id_to_index(id)
        self.positions[:, index] = pos

    @property
    def position_generator(self):  # "generator" is a misleading name, has no yield statement
        def gen(i):
            return self.positions.T[i]
        return gen

    def _get_cell_initial_value(self, id, variable):
        assert isinstance(self.initial_values[variable], LazyArray)
        index = self.id_to_local_index(id)
        return self.initial_values[variable][index]

    def _set_cell_initial_value(self, id, variable, value):
        assert isinstance(self.initial_values[variable], LazyArray)
        index = self.id_to_local_index(id)
        self.initial_values[variable][index] = value

    def nearest(self, position):
        """Return the neuron closest to the specified position."""
        # doesn't always work correctly if a position is equidistant between
        # two neurons, i.e. 0.5 should be rounded up, but it isn't always.
        # also doesn't take account of periodic boundary conditions
        pos = numpy.array([position] * self.positions.shape[1]).transpose()
        dist_arr = (self.positions - pos)**2
        distances = dist_arr.sum(axis=0)
        nearest = distances.argmin()
        return self[nearest]

    def sample(self, n, rng=None):
        """
        Randomly sample `n` cells from the Population, and return a
        PopulationView object.
        """
        assert isinstance(n, int)
        if not rng:
            rng = random.NumpyRNG()
        indices = rng.permutation(numpy.arange(len(self), dtype=numpy.int))[0:n]
        logger.debug("The %d cells selected have indices %s" % (n, indices))
        logger.debug("%s.sample(%s)", self.label, n)
        return self._get_view(indices)

    def get(self, parameter_names, gather=False, simplify=True):
        """
        Get the values of the given parameters for every local cell in the
        population, or, if gather=True, for all cells in the population.
        
        Values will be expressed in the standard PyNN units (i.e. millivolts,
        nanoamps, milliseconds, microsiemens, nanofarads, event per second).
        """
        # if all the cells have the same value for a parameter, should
        # we return just the number, rather than an array?
        if isinstance(parameter_names, basestring):
            parameter_names = (parameter_names,)
            return_list = False
        else:
            return_list = True
        if isinstance(self.celltype, standardmodels.StandardCellType):
            if any(name in self.celltype.computed_parameters() for name in parameter_names):
                native_names = self.celltype.get_native_names() # need all parameters in order to calculate values
            else:
                native_names = self.celltype.get_native_names(*parameter_names)
            native_parameter_space = self._get_parameters(*native_names)
            parameter_space = self.celltype.reverse_translate(native_parameter_space)
        else:
            parameter_space = self._get_parameters(*self.celltype.get_parameter_names())
        parameter_space.evaluate(simplify=simplify) # what if parameter space is homogeneous on some nodes but not on others?

        parameters = dict(parameter_space.items())
        if gather == True and self._simulator.state.num_processes > 1:
            # seems inefficient to do it in a loop - should do as single operation
            for name in parameter_names:
                values = parameter_space[name]
                all_values  = { self._simulator.state.mpi_rank: values.tolist() }
                all_indices = { self._simulator.state.mpi_rank: self.local_cells.tolist() }
                all_values  = recording.gather_dict(all_values)
                all_indices = recording.gather_dict(all_indices)
                if self._simulator.state.mpi_rank == 0:
                    values  = reduce(operator.add, all_values.values())
                    indices = reduce(operator.add, all_indices.values())
                idx    = numpy.argsort(indices)
                values = numpy.array(values)[idx]
            parameters[name] = values
        try:
            values = [parameters[name] for name in parameter_names]
        except KeyError as err:
            raise errors.NonExistentParameterError("%s. Valid parameters for %s are: %s" % (
                err, self.celltype, self.celltype.get_parameter_names()))
        if return_list:
            return values
        else:
            assert len(parameter_names) == 1
            return values[0]

    def set(self, **parameters):
        """
        Set one or more parameters for every cell in the population.

        Values passed to set() may be:
            (1) single values
            (2) RandomDistribution objects
            (3) lists/arrays of values of the same size as the population
            (4) mapping functions, where a mapping function accepts a single
                argument (the cell index) and returns a single value.

        Here, a "single value" may be either a single number or a list/array of
        numbers (e.g. for spike times). Values should be expressed in the
        standard PyNN units (i.e. millivolts, nanoamps, milliseconds,
        microsiemens, nanofarads, event per second).

        Examples::

            p.set(tau_m=20.0, v_rest=-65).
            p.set(spike_times=[0.3, 0.7, 0.9, 1.4])
            p.set(cm=rand_distr, tau_m=lambda i: 10 + i/10.0)
        """
        # TODO: add example using of function of (x,y,z) and Population.position_generator
        if (isinstance(self.celltype, standardmodels.StandardCellType)
            and any(name in self.celltype.computed_parameters() for name in parameters)):
            # need to get existing parameter space of models so we can perform calculations
            native_names = self.celltype.get_native_names()
            parameter_space = self.celltype.reverse_translate(self._get_parameters(*native_names))
            parameter_space.update(**parameters)
        else:
            parameter_space = ParameterSpace(parameters,
                                             self.celltype.get_schema(),
                                             (self.size,),
                                             self.celltype.__class__)
        if isinstance(self.celltype, standardmodels.StandardCellType):
            parameter_space = self.celltype.translate(parameter_space)
        assert parameter_space.shape == (self.size,)
        self._set_parameters(parameter_space)

    @deprecated("set(parametername=value_array)")
    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        self.set(**{parametername: value_array})

    @deprecated("set(parametername=rand_distr)")
    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        # Note that we generate enough random numbers for all cells on all nodes
        # but use only those relevant to this node. This ensures that the
        # sequence of random numbers does not depend on the number of nodes,
        # provided that the same rng with the same seed is used on each node.
        self.set(**{parametername: rand_distr})

    def initialize(self, **initial_values):
        """
        Set initial values of state variables, e.g. the membrane potential.

        Values passed to initialize() may be:
            (1) single numeric values (all neurons set to the same value)
            (2) RandomDistribution objects
            (3) lists/arrays of numbers of the same size as the population
            (4) mapping functions, where a mapping function accepts a single
                argument (the cell index) and returns a single number.

        Values should be expressed in the standard PyNN units (i.e. millivolts,
        nanoamps, milliseconds, microsiemens, nanofarads, event per second).

        Examples::

            p.initialize(v=-70.0)
            p.initialize(v=rand_distr, gsyn_exc=0.0)
            p.initialize(v=lambda i: -65 + i/10.0)
        """
        for variable, value in initial_values.items():
            logger.debug("In Population '%s', initialising %s to %s" % (self.label, variable, value))
            initial_value = LazyArray(value, shape=(self.size,), dtype=float)
            self._set_initial_value_array(variable, initial_value)
            self.initial_values[variable] = initial_value

    def find_units(self, variable):
        return self.celltype.units[variable]

    def can_record(self, variable):
        """Determine whether `variable` can be recorded from this population."""
        return self.celltype.can_record(variable)

    def record(self, variables, to_file=None, sampling_interval=None):
        """
        Record the specified variable or variables for all cells in the
        Population or view.

        `variables` may be either a single variable name or a list of variable
        names. For a given celltype class, `celltype.recordable` contains a list of
        variables that can be recorded for that celltype.

        If specified, `to_file` should be a Neo IO instance and `write_data()`
        will be automatically called when `end()` is called.
        
        `sampling_interval` should be a value in milliseconds, and an integer
        multiple of the simulation timestep.
        """
        if variables is None: # reset the list of things to record
                              # note that if record(None) is called on a view of a population
                              # recording will be reset for the entire population, not just the view
            self.recorder.reset()
        else:
            logger.debug("%s.record('%s')", self.label, variables)
            if self._record_filter is None:
                self.recorder.record(variables, self.all_cells, sampling_interval)
            else:
                self.recorder.record(variables, self._record_filter, sampling_interval)
        if isinstance(to_file, basestring):
            self.recorder.file = to_file

    @deprecated("record('v')")
    def record_v(self, to_file=True):
        """
        Record the membrane potential for all cells in the Population.
        """
        self.record('v', to_file)

    @deprecated("record(['gsyn_exc', 'gsyn_inh'])")
    def record_gsyn(self, to_file=True):
        """
        Record synaptic conductances for all cells in the Population.
        """
        self.record(['gsyn_exc', 'gsyn_inh'], to_file)

    def write_data(self, io, variables='all', gather=True, clear=False, annotations=None):
        """
        Write recorded data to file, using one of the file formats supported by
        Neo.

        `io`:
            a Neo IO instance
        `variables`:
            either a single variable name or a list of variable names.
            Variables must have been previously recorded, otherwise an
            Exception will be raised.

        For parallel simulators, if `gather` is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only data from the cells
        simulated on that node.

        If `clear` is True, recorded data will be deleted from the `Population`.
        
        `annotations` should be a dict containing simple data types such as
        numbers and strings. The contents will be written into the output data
        file as metadata.
        """
        logger.debug("Population %s is writing %s to %s [gather=%s, clear=%s]" % (self.label, variables, io, gather, clear))
        self.recorder.write(variables, io, gather, self._record_filter, clear=clear,
                            annotations=annotations)

    def get_data(self, variables='all', gather=True, clear=False):
        """
        Return a Neo `Block` containing the data (spikes, state variables)
        recorded from the Population.

        `variables` - either a single variable name or a list of variable names
                      Variables must have been previously recorded, otherwise an
                      Exception will be raised.

        For parallel simulators, if `gather` is True, all data will be gathered
        to all nodes and the Neo `Block` will contain data from all nodes.
        Otherwise, the Neo `Block` will contain only data from the cells
        simulated on the local node.

        If `clear` is True, recorded data will be deleted from the `Population`.
        """
        return self.recorder.get(variables, gather, self._record_filter, clear)

    @deprecated("write_data(file, 'spikes')")
    def printSpikes(self, file, gather=True, compatible_output=True):
        self.write_data(file, 'spikes', gather)

    @deprecated("get_data('spikes')")
    def getSpikes(self, gather=True, compatible_output=True):
        return self.get_data('spikes', gather)

    @deprecated("write_data(file, 'v')")
    def print_v(self, file, gather=True, compatible_output=True):
        self.write_data(file, 'v', gather)

    @deprecated("get_data('v')")
    def get_v(self, gather=True, compatible_output=True):
        return self.get_data('v', gather)

    @deprecated("write_data(file, ['gsyn_exc', 'gsyn_inh'])")
    def print_gsyn(self, file, gather=True, compatible_output=True):
        self.write_data(file, ['gsyn_exc', 'gsyn_inh'], gather)

    @deprecated("get_data(['gsyn_exc', 'gsyn_inh'])")
    def get_gsyn(self, gather=True, compatible_output=True):
        return self.get_data(['gsyn_exc', 'gsyn_inh'], gather)

    def get_spike_counts(self, gather=True):
        """
        Returns a dict containing the number of spikes for each neuron.
        
        The dict keys are neuron IDs, not indices.
        """
        # arguably, we should use indices
        return self.recorder.count('spikes', gather, self._record_filter)

    @deprecated("mean_spike_count()")
    def meanSpikeCount(self, gather=True):
        return self.mean_spike_count(gather)

    def mean_spike_count(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        spike_counts = self.get_spike_counts(gather)
        total_spikes = sum(spike_counts.values())
        if self._simulator.state.mpi_rank == 0 or not gather:  # should maybe use allgather, and get the numbers on all nodes
            if len(spike_counts) > 0:
                return float(total_spikes)/len(spike_counts)
            else:
                return 0
        else:
            return numpy.nan

    def inject(self, current_source):
        """
        Connect a current source to all cells in the Population.
        """
        if not self.celltype.injectable:
            raise TypeError("Can't inject current into a spike source.")
        current_source.inject_into(self)

    # name should be consistent with saving/writing data, i.e. save_data() and save_positions() or write_data() and write_positions()
    def save_positions(self, file):
        """
        Save positions to file. The output format is ``id x y z``
        """
        # first column should probably be indices, not ids. This would make it
        # simulator independent.
        if isinstance(file, basestring):
            file = recording.files.StandardTextFile(file, mode='w')
        cells  = self.all_cells
        result = numpy.empty((len(cells), 4))
        result[:,0]   = cells
        result[:,1:4] = self.positions.T
        if self._simulator.state.mpi_rank == 0:
            file.write(result, {'population' : self.label})
            file.close()


class Population(BasePopulation):
    """
    A group of neurons all of the same type. "Population" is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.

    Arguments:
        `size`:
            number of cells in the Population. For backwards-compatibility,
            `size` may also be a tuple giving the dimensions of a grid,
            e.g. ``size=(10,10)`` is equivalent to ``size=100`` with ``structure=Grid2D()``.

        `cellclass`:
            a cell type (a class inheriting from :class:`pyNN.models.BaseCellType`).

        `cellparams`:
            a dict, or other mapping, containing parameters, which is passed to
            the neuron model constructor.

        `structure`:
            a :class:`pyNN.space.Structure` instance, used to specify the
            positions of neurons in space.

        `initial_values`:
            a dict, or other mapping, containing initial values for the neuron
            state variables.

        `label`:
            a name for the population. One will be auto-generated if this is not
            supplied.
    """
    _nPop = 0

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        """
        Create a population of neurons all of the same type.
        """
        if not isinstance(size, int):  # also allow a single integer, for a 1D population
            assert isinstance(size, tuple), "`size` must be an integer or a tuple of ints. You have supplied a %s" % type(size)
            # check the things inside are ints
            for e in size:
                assert isinstance(e, int), "`size` must be an integer or a tuple of ints. Element '%s' is not an int" % str(e)

            assert structure is None, "If you specify `size` as a tuple you may not specify structure."
            if len(size) == 1:
                structure = space.Line()
            elif len(size) == 2:
                nx, ny = size
                structure = space.Grid2D(nx/float(ny))
            elif len(size) == 3:
                nx, ny, nz = size
                structure = space.Grid3D(nx/float(ny), nx/float(nz))
            else:
                raise Exception("A maximum of 3 dimensions is allowed. What do you think this is, string theory?")
            size = int(reduce(operator.mul, size)) # NEST doesn't like numpy.int, so to be safe we cast to Python int
        self.size = size
        self.label = label or 'population%d' % Population._nPop
        self._structure = structure or space.Line()
        self._positions = None
        self._is_sorted = True
        if isinstance(cellclass, BaseCellType):
            self.celltype = cellclass
            assert cellparams is None   # cellparams being retained for backwards compatibility, but use is deprecated
        elif issubclass(cellclass, BaseCellType):
            self.celltype = cellclass(**cellparams)
            # emit deprecation warning
        else:
            raise TypeError("cellclass must be an instance or subclass of BaseCellType, not a %s" % type(cellclass))
        self.annotations = {}
        self.recorder = self._recorder_class(self)
        # Build the arrays of cell ids
        # Cells on the local node are represented as ID objects, other cells by integers
        # All are stored in a single numpy array for easy lookup by address
        # The local cells are also stored in a list, for easy iteration
        self._create_cells()
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        self.initial_values = {}
        all_initial_values = self.celltype.default_initial_values.copy()
        all_initial_values.update(initial_values)
        self.initialize(**all_initial_values)
        Population._nPop += 1

    def __repr__(self):
        return "Population(%d, %r, structure=%r, label=%r)" % (self.size, self.celltype, self.structure, self.label)

    @property
    def local_cells(self):
        """
        An array containing cell ids for the local node.
        """
        return self.all_cells[self._mask_local]

    def id_to_index(self, id):
        """
        Given the ID(s) of cell(s) in the Population, return its (their) index
        (order in the Population).

            >>> assert p.id_to_index(p[5]) == 5
        """
        if not numpy.iterable(id):
            if not self.first_id <= id <= self.last_id:
                raise ValueError("id should be in the range [%d,%d], actually %d" % (self.first_id, self.last_id, id))
            return int(id - self.first_id)  # this assumes ids are consecutive
        else:
            if isinstance(id, PopulationView):
                id = id.all_cells
            id = numpy.array(id)
            if (self.first_id > id.min()) or (self.last_id < id.max()):
                raise ValueError("ids should be in the range [%d,%d], actually [%d, %d]" % (self.first_id, self.last_id, id.min(), id.max()))
            return (id - self.first_id).astype(numpy.int)  # this assumes ids are consecutive

    def id_to_local_index(self, id):
        """
        Given the ID(s) of cell(s) in the Population, return its (their) index
        (order in the Population), counting only cells on the local MPI node.
        """
        if self._simulator.state.num_processes > 1:
            return self.local_cells.tolist().index(id)          # probably very slow
            #return numpy.nonzero(self.local_cells == id)[0][0] # possibly faster?
            # another idea - get global index, use idx-sum(mask_local[:idx])?
        else:
            return self.id_to_index(id)

    def _get_structure(self):
        """The spatial structure of the Population."""
        return self._structure

    def _set_structure(self, structure):
        assert isinstance(structure, space.BaseStructure)
        if self._structure is None or structure != self._structure:
            self._positions = None  # setting a new structure invalidates previously calculated positions
            self._structure = structure
    structure = property(fget=_get_structure, fset=_set_structure)
    # arguably structure should be read-only, i.e. it is not possible to change it after Population creation

    def _get_positions(self):
        """
        Try to return self._positions. If it does not exist, create it and then
        return it.
        """
        if self._positions is None:
            self._positions = self.structure.generate_positions(self.size)
        assert self._positions.shape == (3, self.size)
        return self._positions

    def _set_positions(self, pos_array):
        assert isinstance(pos_array, numpy.ndarray)
        assert pos_array.shape == (3, self.size), "%s != %s" % (pos_array.shape, (3, self.size))
        self._positions = pos_array.copy()  # take a copy in case pos_array is changed later
        self._structure = None  # explicitly setting positions destroys any previous structure

    positions = property(_get_positions, _set_positions,
                         doc="""A 3xN array (where N is the number of neurons in the Population)
                         giving the x,y,z coordinates of all the neurons (soma, in the
                         case of non-point models).""")

    def annotate(self, **annotations):
        self.annotations.update(annotations)

    def describe(self, template='population_default.txt', engine='default'):
        """
        Returns a human-readable description of the population.

        The output may be customized by specifying a different template
        together with an associated template engine (see :mod:`pyNN.descriptions`).

        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {
            "label": self.label,
            "celltype": self.celltype.describe(template=None),
            "structure": None,
            "size": self.size,
            "size_local": len(self.local_cells),
            "first_id": self.first_id,
            "last_id": self.last_id,
        }
        context.update(self.annotations)
        if len(self.local_cells) > 0:
            first_id = self.local_cells[0]
            context.update({
                "local_first_id": first_id,
                "cell_parameters": {} #first_id.get_parameters(),
            })
        if self.structure:
            context["structure"] = self.structure.describe(template=None)
        return descriptions.render(engine, template, context)


class PopulationView(BasePopulation):
    """
    A view of a subset of neurons within a Population.

    In most ways, Populations and PopulationViews have the same behaviour, i.e.
    they can be recorded, connected with Projections, etc. It should be noted
    that any changes to neurons in a PopulationView will be reflected in the
    parent Population and vice versa.

    It is possible to have views of views.

    Arguments:
        selector:
            a slice or numpy mask array. The mask array should either be a
            boolean array of the same size as the parent, or an integer array
            containing cell indices, i.e. if p.size == 5::

                PopulationView(p, array([False, False, True, False, True]))
                PopulationView(p, array([2,4]))
                PopulationView(p, slice(2,5,2))

            will all create the same view.
    """

    def __init__(self, parent, selector, label=None):
        """
        Create a view of a subset of neurons within a parent Population or
        PopulationView.
        """
        self.parent = parent
        self.mask = selector # later we can have fancier selectors, for now we just have numpy masks
        # maybe just redefine __getattr__ instead of the following...
        self.celltype     = self.parent.celltype
        # If the mask is a slice, IDs will be consecutives without duplication.
        # If not, then we need to remove duplicated IDs
        if not isinstance(self.mask, slice):
            if isinstance(self.mask, list):
                self.mask = numpy.array(self.mask)
            if self.mask.dtype is numpy.dtype('bool'):
                if len(self.mask) != len(self.parent):
                    raise Exception("Boolean masks should have the size of Parent Population")
                self.mask = numpy.arange(len(self.parent))[self.mask]
            if len(numpy.unique(self.mask)) != len(self.mask):
                logging.warning("PopulationView can contain only once each ID, duplicated IDs are remove")
                self.mask = numpy.unique(self.mask)
        self.all_cells    = self.parent.all_cells[self.mask]  # do we need to ensure this is ordered?
        idx = numpy.argsort(self.all_cells)
        self._is_sorted =  numpy.all(idx == numpy.arange(len(self.all_cells)))
        self.size         = len(self.all_cells)
        self.label  = label or "view of '%s' with size %s" % (parent.label, self.size)
        self._mask_local  = self.parent._mask_local[self.mask]
        self.local_cells  = self.all_cells[self._mask_local]
        self.first_id     = numpy.min(self.all_cells) # only works if we assume all_cells is sorted, otherwise could use min()
        self.last_id      = numpy.max(self.all_cells)
        self.recorder    = self.parent.recorder
        self._record_filter= self.all_cells

    def __repr__(self):
        return "PopulationView(parent=%r, selector=%r, label=%r)" % (self.parent, self.mask, self.label)

    @property
    def initial_values(self):
        # this is going to be complex - if we keep initial_values as a dict,
        # need to return a dict-like object that takes account of self.mask
        raise NotImplementedError

    @property
    def structure(self):
        """The spatial structure of the parent Population."""
        return self.parent.structure
    # should we allow setting structure for a PopulationView? Maybe if the
    # parent has some kind of CompositeStructure?

    @property
    def positions(self):
        return self.parent.positions.T[self.mask].T  # make positions N,3 instead of 3,N to avoid all this transposing?

    def id_to_index(self, id):
        """
        Given the ID(s) of cell(s) in the PopulationView, return its/their
        index/indices (order in the PopulationView).

            >>> assert pv.id_to_index(pv[3]) == 3
        """
        if not numpy.iterable(id):
            if self._is_sorted:
                if id not in self.all_cells:
                    raise IndexError("ID %s not present in the View" %id)
                return numpy.searchsorted(self.all_cells, id)
            else:
                result = numpy.where(self.all_cells == id)[0]
            if len(result) == 0:
                raise IndexError("ID %s not present in the View" %id)
            else:
                return result
        else:
            if self._is_sorted:
                return numpy.searchsorted(self.all_cells, id)
            else:
                result = numpy.array([], dtype=numpy.int)
                for item in id:
                    data = numpy.where(self.all_cells == item)[0]
                    if len(data) == 0:
                        raise IndexError("ID %s not present in the View" %item)
                    elif len(data) > 1:
                        raise Exception("ID %s is duplicated in the View" %item)
                    else:
                        result = numpy.append(result, data)
                return result

    @property
    def grandparent(self):
        """
        Returns the parent Population at the root of the tree (since the
        immediate parent may itself be a PopulationView).

        The name "grandparent" is of course a little misleading, as it could
        be just the parent, or the great, great, great, ..., grandparent.
        """
        if hasattr(self.parent, "parent"):
            return self.parent.grandparent
        else:
            return self.parent

    def index_in_grandparent(self, indices):
        """
        Given an array of indices, return the indices in the parent population
        at the root of the tree.
        """
        indices_in_parent = numpy.arange(self.parent.size)[self.mask][indices]
        if hasattr(self.parent, "parent"):
            return self.parent.index_in_grandparent(indices_in_parent)
        else:
            return indices_in_parent

    def describe(self, template='populationview_default.txt', engine='default'):
        """
        Returns a human-readable description of the population view.

        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).

        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {"label": self.label,
                   "parent": self.parent.label,
                   "mask": self.mask,
                   "size": self.size}
        return descriptions.render(engine, template, context)



class Assembly(object):
    """
    A group of neurons, may be heterogeneous, in contrast to a Population where
    all the neurons are of the same type.

    Arguments:
        populations:
            Populations or PopulationViews
        kwargs:
            May contain a keyword argument 'label'
    """
    _count = 0

    def __init__(self, *populations, **kwargs):
        """
        Create an Assembly of Populations and/or PopulationViews.
        """
        if kwargs:
            assert list(kwargs.keys()) == ['label']
        self.populations = []
        for p in populations:
            self._insert(p)
        self.label = kwargs.get('label', 'assembly%d' % Assembly._count)
        assert isinstance(self.label, basestring), "label must be a string or unicode"
        Assembly._count += 1

    def __repr__(self):
        return "Assembly(*%r, label=%r)" % (self.populations, self.label)

    def _insert(self, element):
        if not isinstance(element, BasePopulation):
            raise TypeError("argument is a %s, not a Population." % type(element).__name__)
        if isinstance(element, PopulationView):
            if not element.parent in self.populations:
                double = False
                for p in self.populations:
                    data = numpy.concatenate((p.all_cells, element.all_cells))
                    if len(numpy.unique(data))!= len(p.all_cells) + len(element.all_cells):
                        logging.warning('Adding a PopulationView to an Assembly containing elements already present is not posible')
                        double = True #Should we automatically remove duplicated IDs ?
                        break
                if not double:
                    self.populations.append(element)
            else:
                logging.warning('Adding a PopulationView to an Assembly when parent Population is there is not possible')
        elif isinstance(element, BasePopulation):
            if not element in self.populations:
                self.populations.append(element)
            else:
                logging.warning('Adding a Population twice in an Assembly is not possible')

    @property
    def local_cells(self):
        result = self.populations[0].local_cells
        for p in self.populations[1:]:
            result = numpy.concatenate((result, p.local_cells))
        return result

    @property
    def all_cells(self):
        result = self.populations[0].all_cells
        for p in self.populations[1:]:
            result = numpy.concatenate((result, p.all_cells))
        return result

    def all(self):
        """Iterator over cell ids on all nodes."""
        return iter(self.all_cells)

    @property
    def _is_sorted(self):
        idx = numpy.argsort(self.all_cells)
        return numpy.all(idx == numpy.arange(len(self.all_cells)))

    @property
    def _homogeneous_synapses(self):
        cb = [p.celltype.conductance_based for p in self.populations]
        return all(cb) or not any(cb)

    @property
    def conductance_based(self):
        """
        `True` if the post-synaptic response is modelled as a change
        in conductance, `False` if a change in current.
        """
        return all(p.celltype.conductance_based for p in self.populations)

    @property
    def receptor_types(self):
        """
        Return a list of receptor types that are common to all populations
        within the assembly.
        """
        rts = set(self.populations[0].celltype.receptor_types)
        if len(self.populations) > 1:
            for p in self.populations[1:]:
                rts = rts.intersection(set(p.celltype.receptor_types))
        return rts

    def find_units(self, variable):
        units = set(p.find_units(variable) for p in self.populations)
        if len(units) > 1:
            raise ValueError("Inconsistent units")
        return units

    @property
    def _mask_local(self):
        result = self.populations[0]._mask_local
        for p in self.populations[1:]:
            result = numpy.concatenate((result, p._mask_local))
        return result

    @property
    def first_id(self):
        return numpy.min(self.all_cells)

    @property
    def last_id(self):
        return numpy.max(self.all_cells)

    def id_to_index(self, id):
        """
        Given the ID(s) of cell(s) in the Assembly, return its (their) index
        (order in the Assembly)::

            >>> assert p.id_to_index(p[5]) == 5
            >>> assert p.id_to_index(p.index([1,2,3])) == [1,2,3]
        """
        all_cells = self.all_cells
        if not numpy.iterable(id):
            if self._is_sorted:
                return numpy.searchsorted(all_cells, id)
            else:
                result = numpy.where(all_cells == id)[0]
            if len(result) == 0:
                raise IndexError("ID %s not present in the View" %id)
            else:
                return result
        else:
            if self._is_sorted:
                return numpy.searchsorted(all_cells, id)
            else:
                result = numpy.array([], dtype=numpy.int)
                for item in id:
                    data = numpy.where(all_cells == item)[0]
                    if len(data) == 0:
                        raise IndexError("ID %s not present in the Assembly" %item)
                    elif len(data) > 1:
                        raise Exception("ID %s is duplicated in the Assembly" %item)
                    else:
                        result = numpy.append(result, data)
                return result

    @property
    def positions(self):
        result = self.populations[0].positions
        for p in self.populations[1:]:
            result = numpy.hstack((result, p.positions))
        return result

    @property
    def size(self):
        return sum(p.size for p in self.populations)

    def __iter__(self):
        """
        Iterator over cells in all populations within the Assembly, for cells
        on the local MPI node.
        """
        iterators = [iter(p) for p in self.populations]
        return chain(*iterators)

    def __len__(self):
        """Return the total number of cells in the population (all nodes)."""
        return self.size

    def __getitem__(self, index):
        """
        Where `index` is an integer, return an ID.
        Where `index` is a slice, tuple, list or numpy array, return a new Assembly
        consisting of appropriate populations and (possibly newly created)
        population views.
        """
        count = 0; boundaries = [0]
        for p in self.populations:
            count += p.size
            boundaries.append(count)
        boundaries = numpy.array(boundaries, dtype=numpy.int)

        if isinstance(index, (int, numpy.integer)): # return an ID
            pindex = boundaries[1:].searchsorted(index, side='right')
            return self.populations[pindex][index-boundaries[pindex]]
        elif isinstance(index, (slice, tuple, list, numpy.ndarray)):
            if isinstance(index, slice):
                indices = numpy.arange(self.size)[index]
            else:
                indices = numpy.array(index)
            pindices = boundaries[1:].searchsorted(indices, side='right')
            views = [self.populations[i][indices[pindices==i] - boundaries[i]] for i in numpy.unique(pindices)]
            return self.__class__(*views)
        else:
            raise TypeError("indices must be integers, slices, lists, arrays, not %s" % type(index).__name__)

    def __add__(self, other):
        """
        An Assembly may be added to a Population, PopulationView or Assembly
        with the '+' operator, returning a new Assembly, e.g.::

            a2 = a1 + p
        """
        if isinstance(other, BasePopulation):
            return self.__class__(*(self.populations + [other]))
        elif isinstance(other, Assembly):
            return self.__class__(*(self.populations + other.populations))
        else:
            raise TypeError("can only add a Population or another Assembly to an Assembly")

    def __iadd__(self, other):
        """
        A Population, PopulationView or Assembly may be added to an existing
        Assembly using the '+=' operator, e.g.::

            a += p
        """
        if isinstance(other, BasePopulation):
            self._insert(other)
        elif isinstance(other, Assembly):
            for p in other.populations:
                self._insert(p)
        else:
            raise TypeError("can only add a Population or another Assembly to an Assembly")
        return self

    def sample(self, n, rng=None):
        """
        Randomly sample `n` cells from the Assembly, and return a Assembly
        object.
        """
        assert isinstance(n, int)
        if not rng:
            rng = random.NumpyRNG()
        indices = rng.permutation(numpy.arange(len(self), dtype=numpy.int))[0:n]
        logger.debug("The %d cells recorded have indices %s" % (n, indices))
        logger.debug("%s.sample(%s)", self.label, n)
        return self[indices]

    def initialize(self, **initial_values):
        """
        Set the initial values of the state variables of the neurons in
        this assembly.
        """
        for p in self.populations:
            p.initialize(**initial_values)

    def get(self, parameter_names, gather=False, simplify=True):
        """
        Get the values of the given parameters for every local cell in the
        Assembly, or, if gather=True, for all cells in the Assembly.
        """
        if isinstance(parameter_names, basestring):
            parameter_names = (parameter_names,)
            return_list = False
        else:
            return_list = True

        parameters = defaultdict(list)
        for p in self.populations:
            population_values = p.get(parameter_names, gather, simplify=False)
            for name, arr in zip(parameter_names, population_values):
                parameters[name].append(arr)
        for name, value_list in parameters.items():
            parameters[name] = numpy.hstack(value_list)
            if simplify:
                parameters[name] = simplify_parameter_array(parameters[name])
        values = [parameters[name] for name in parameter_names]
        if return_list:
            return values
        else:
            assert len(parameter_names) == 1
            return values[0]


    def set(self, **parameters):
        """
        Set one or more parameters for every cell in the Assembly.

        Values passed to set() may be:
            (1) single values
            (2) RandomDistribution objects
            (3) mapping functions, where a mapping function accepts a single
                argument (the cell index) and returns a single value.

        Here, a "single value" may be either a single number or a list/array of
        numbers (e.g. for spike times).
        """
        for p in self.populations:
            p.set(**parameters)

    @deprecated("set(parametername=rand_distr)")
    def rset(self, parametername, rand_distr):
        self.set(parametername=rand_distr)

    def record(self, variables, to_file=None, sampling_interval=None):
        """
        Record the specified variable or variables for all cells in the Assembly.

        `variables` may be either a single variable name or a list of variable
        names. For a given celltype class, `celltype.recordable` contains a list of
        variables that can be recorded for that celltype.

        If specified, `to_file` should be a Neo IO instance and `write_data()`
        will be automatically called when `end()` is called.
        """
        for p in self.populations:
            p.record(variables, to_file, sampling_interval)

    @deprecated("record('v')")
    def record_v(self, to_file=True):
        """Record the membrane potential from all cells in the Assembly."""
        self.record('v', to_file)

    @deprecated("record(['gsyn_exc', 'gsyn_inh'])")
    def record_gsyn(self, to_file=True):
        """Record synaptic conductances from all cells in the Assembly."""
        self.record(['gsyn_exc', 'gsyn_inh'], to_file)

    def get_population(self, label):
        """
        Return the Population/PopulationView from within the Assembly that has
        the given label. If no such Population exists, raise KeyError.
        """
        for p in self.populations:
            if label == p.label:
                return p
        raise KeyError("Assembly does not contain a population with the label %s" % label)

    def save_positions(self, file):
        """
        Save positions to file. The output format is id x y z
        """
        # this should be rewritten to use self.positions and recording.files
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        cells  = self.all_cells
        result = numpy.empty((len(cells), 4))
        result[:,0]   = cells
        result[:,1:4] = self.positions.T
        if self._simulator.state.mpi_rank == 0:
            file.write(result, {'assembly' : self.label})
            file.close()

    @property
    def position_generator(self):
        def gen(i):
            return self.positions[:,i]
        return gen

    def get_data(self, variables='all', gather=True, clear=False, annotations=None):
        """
        Return a Neo `Block` containing the data (spikes, state variables)
        recorded from the Assembly.

        `variables` - either a single variable name or a list of variable names
                      Variables must have been previously recorded, otherwise an
                      Exception will be raised.

        For parallel simulators, if `gather` is True, all data will be gathered
        to all nodes and the Neo `Block` will contain data from all nodes.
        Otherwise, the Neo `Block` will contain only data from the cells
        simulated on the local node.

        If `clear` is True, recorded data will be deleted from the `Assembly`.
        """
        name = self.label
        description = self.describe()
        blocks = [p.get_data(variables, gather, clear) for p in self.populations]
        offset = 0
        for block,p in zip(blocks, self.populations):
            for segment in block.segments:
                #segment.name = name
                #segment.description = description
                for signal_array in segment.analogsignalarrays:
                    signal_array.channel_index = numpy.array(signal_array.channel_index) + offset  # hack
            offset += p.size
        for i,block in enumerate(blocks): ##
            logger.debug("%d: %s", i, block.name)
            for j,segment in enumerate(block.segments):
                logger.debug("  %d: %s", j, segment.name)
                for arr in segment.analogsignalarrays:
                    logger.debug("    %s %s", arr.shape, arr.name)
        merged_block = blocks[0]
        for block in blocks[1:]:
            merged_block.merge(block)
        merged_block.name = name
        merged_block.description = description
        if annotations:
            merged_block.annotate(**annotations)
        return merged_block

    @deprecated("get_data('spikes')")
    def getSpikes(self, gather=True, compatible_output=True):
        return self.get_data('spikes', gather)

    @deprecated("get_data('v')")
    def get_v(self, gather=True, compatible_output=True):
        return self.get_data('v', gather)

    @deprecated("get_data(['gsyn_exc', 'gsyn_inh'])")
    def get_gsyn(self, gather=True, compatible_output=True):
        return self.get_data(['gsyn_exc', 'gsyn_inh'], gather)

    def mean_spike_count(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        spike_counts = self.get_spike_counts()
        total_spikes = sum(spike_counts.values())
        if self._simulator.state.mpi_rank() == 0 or not gather:  # should maybe use allgather, and get the numbers on all nodes
            return float(total_spikes)/len(spike_counts)
        else:
            return numpy.nan

    def get_spike_counts(self, gather=True):
        """
        Returns the number of spikes for each neuron.
        """
        try:
            spike_counts = self.populations[0].recorders['spikes'].count(gather, self.populations[0]._record_filter)
        except errors.NothingToWriteError:
            spike_counts = {}
        for p in self.populations[1:]:
            try:
                spike_counts.update(p.recorders['spikes'].count(gather, p._record_filter))
            except errors.NothingToWriteError:
                pass
        return spike_counts

    def write_data(self, io, variables='all', gather=True, clear=False, annotations=None):
        """
        Write recorded data to file, using one of the file formats supported by
        Neo.

        `io`:
            a Neo IO instance
        `variables`:
            either a single variable name or a list of variable names.
            Variables must have been previously recorded, otherwise an
            Exception will be raised.

        For parallel simulators, if `gather` is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only data from the cells
        simulated on that node.

        If `clear` is True, recorded data will be deleted from the `Population`.
        """
        if isinstance(io, basestring):
            io = recording.get_io(io)
        if gather == False and self._simulator.state.num_processes > 1:
            io.filename += '.%d' % self._simulator.state.mpi_rank
        logger.debug("Recorder is writing '%s' to file '%s' with gather=%s" % (
                                               variables, io.filename, gather))
        data = self.get_data(variables, gather, clear, annotations)
        if self._simulator.state.mpi_rank == 0 or gather == False:
            logger.debug("Writing data to file %s" % io)
            io.write(data)

    @deprecated("write_data(file, 'spikes')")
    def printSpikes(self, file, gather=True, compatible_output=True):
        self.write_data(file, 'spikes', gather)

    @deprecated("write_data(file, 'v')")
    def print_v(self, file, gather=True, compatible_output=True):
        self.write_data(file, 'v', gather)

    @deprecated("write_data(['gsyn_exc', 'gsyn_inh'])")
    def print_gsyn(self, file, gather=True, compatible_output=True):
        self.write_data(file, ['gsyn_exc', 'gsyn_inh'], gather)

    def inject(self, current_source):
        """
        Connect a current source to all cells in the Assembly.
        """
        for p in self.populations:
            current_source.inject_into(p)

    def describe(self, template='assembly_default.txt', engine='default'):
        """
        Returns a human-readable description of the assembly.

        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).

        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {"label": self.label,
                   "populations": [p.describe(template=None) for p in self.populations]}
        return descriptions.render(engine, template, context)
