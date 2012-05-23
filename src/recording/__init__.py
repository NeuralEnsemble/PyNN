"""
Defines classes and functions for managing recordings (spikes, membrane
potential etc).

These classes and functions are not part of the PyNN API, and are only for
internal use.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import logging
import os.path
import numpy
import os
from copy import copy
from collections import defaultdict
from pyNN import errors
import neo.io
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = logging.getLogger("PyNN")

numpy1_1_formats = {'spikes': "%g\t%d",
                    'v': "%g\t%g\t%d",
                    'gsyn': "%g\t%g\t%g\t%d"}
numpy1_0_formats = {'spikes': "%g", # only later versions of numpy support different
                    'v': "%g",      # formats for different columns
                    'gsyn': "%g"}

if MPI:
    mpi_comm = MPI.COMM_WORLD
MPI_ROOT = 0

UNITS_MAP = {
    'spikes': 'ms',
    'v': 'mV',
    'gsyn_exc': 'uS',
    'gsyn_inh': 'uS'
}

def rename_existing(filename):
    if os.path.exists(filename):
        os.system('mv %s %s_old' % (filename, filename))
        logger.warning("File %s already exists. Renaming the original file to %s_old" % (filename, filename))

def gather_array(data):
    # gather 1D or 2D numpy arrays
    if MPI is None:
        raise Exception("Trying to gather data without MPI installed. If you are not running a distributed simulation, this is a bug in PyNN.")
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size = data.size
    sizes = mpi_comm.gather(size, root=MPI_ROOT) or []
    # now we pass the data
    displacements = [sum(sizes[:i]) for i in range(len(sizes))]
    #print mpi_comm.rank, sizes, displacements, data
    gdata = numpy.empty(sum(sizes))
    mpi_comm.Gatherv([data.flatten(), size, MPI.DOUBLE],
                     [gdata, (sizes, displacements), MPI.DOUBLE],
                     root=MPI_ROOT)
    if len(data.shape) == 1:
        return gdata
    else:
        num_columns = data.shape[1]
        return gdata.reshape((gdata.size/num_columns, num_columns))



def gather_dict(D):
    # Note that if the same key exists on multiple nodes, the value from the
    # node with the highest rank will appear in the final dict.
    Ds = mpi_comm.gather(D, root=MPI_ROOT)
    if Ds:
        for otherD in Ds:
            D.update(otherD)
    return D


def gather_blocks(data):
    """Gather Neo Blocks"""
    if MPI is None:
        raise Exception("Trying to gather data without MPI installed. If you are not running a distributed simulation, this is a bug in PyNN.")
    assert isinstance(data, neo.Block)
    # for now, use gather_dict, which will probably be slow. Can optimize later
    D = {mpi_comm.rank: data}
    D = gather_dict(D)
    blocks = D.values()
    merged = blocks[0]
    for block in blocks[1:]:
        merged.merge(block)
    return merged


def mpi_sum(x):
    if MPI and mpi_comm.size > 1:
        return mpi_comm.allreduce(x, op=MPI.SUM)
    else:
        return x


def normalize_variables_arg(variables):
    """If variables is a single string, encapsulate it in a list."""
    if isinstance(variables, basestring) and variables != 'all':
        return [variables]
    else:
        return variables

def get_io(filename):
    """
    Return a Neo IO instance, guessing the type based on the filename suffix.
    """
    logger.debug("Creating Neo IO for filename %s" % filename)
    extension = os.path.splitext(filename)[1]
    if extension in ('.txt', '.ras', '.v', '.gsyn'):
        return neo.io.PyNNTextIO(filename=filename)
    elif extension in ('.h5',):
        return neo.io.NeoHdf5IO(filename=filename)
    elif extension in ('.pkl', '.pickle'):
        return neo.io.PickleIO(filename=filename)
    else: # function to be improved later
        raise Exception("file extension %s not supported" % extension)

def filter_by_variables(segment, variables):
    """
    Return a new `Segment` containing only recordings of the variables given in
    the list `variables`
    """
    if variables == 'all':
        return segment
    else:
        new_segment = copy(segment) # shallow copy
        if 'spikes' not in variables:
            new_segment.spiketrains = []
        new_segment.analogsignals = [sig for sig in segment.analogsignals if sig.name in variables]
        # also need to handle Units, RecordingChannels
        return new_segment


class DataCache(object):
    # primitive implementation for now, storing in memory - later can consider caching to disk
    def __init__(self):
        self._data = []

    def __iter__(self):
        return iter(self._data)

    def store(self, obj):
        if obj not in self._data:
            logger.debug("Adding %s to cache" % obj)
            self._data.append(obj)

    def clear(self):
        self._data = []


class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""

    def __init__(self, population, file=None):
        """
        Create a recorder.

        `population` -- the Population instance which is being recorded by the
                        recorder
        `file` -- one of:
            - a file-name,
            - `None` (write to a temporary file)
            - `False` (write to memory).
        """
        self.file = file
        self.population = population # needed for writing header information
        self.recorded = defaultdict(set)
        self.cache = DataCache()
        self._simulator.state.recorders.add(self)
        self.clear_flag = False

    def record(self, variables, ids):
        """
        Add the cells in `ids` to the sets of recorded cells for the given variables.
        """
        logger.debug('Recorder.record(<%d cells>)' % len(ids))
        ids = set([id for id in ids if id.local])
        for variable in normalize_variables_arg(variables):
            if not self.population.can_record(variable):
                raise errors.RecordingError(variable, self.population.celltype)
            new_ids = ids.difference(self.recorded[variable])
            self.recorded[variable] = self.recorded[variable].union(ids)
            self._record(variable, new_ids)

    def reset(self):
        """Reset the list of things to be recorded."""
        self._reset()
        self.recorded = defaultdict(set)

    def filter_recorded(self, variable, filter_ids):
        if filter_ids is not None:
            return set(filter_ids).intersection(self.recorded[variable])
        else:
            return self.recorded[variable]

    def get(self, variables, gather=False, filter_ids=None, clear=False):
        """Return the recorded data as a Neo `Block`."""
        variables = normalize_variables_arg(variables)
        data = neo.Block()
        data.segments = [filter_by_variables(segment, variables)
                         for segment in self.cache]
        if self._simulator.state.running: # reset() has not been called, so current segment is not in cache
            data.segments.append(self._get_current_segment(filter_ids=filter_ids, variables=variables))
        data.name = self.population.label
        data.description = self.population.describe()
        data.rec_datetime = data.segments[0].rec_datetime
        data.annotate(**self.metadata)
        if gather and self._simulator.state.num_processes > 1:
            data = gather_blocks(data)
        if clear:
            self.cache.clear()
        self.clear_flag = True
        return data

    def write(self, variables, file=None, gather=False, filter_ids=None, clear=False):
        """Write recorded data to a Neo IO"""
        if isinstance(file, basestring):
            file = get_io(file)
        io = file or self.file
        if gather==False and self._simulator.state.num_processes > 1:
            io.filename += '.%d' % self._simulator.state.mpi_rank
        logger.debug("Recorder is writing '%s' to file '%s' with gather=%s" % (
                                               variables, io.filename, gather))
        data = self.get(variables, gather, filter_ids, clear)
        if self._simulator.state.mpi_rank == 0 or gather == False:
            # Open the output file, if necessary and write the data
            logger.debug("Writing data to file %s" % io)
            io.write_block(data)

    @property
    def metadata(self):
        metadata = {
                'size': self.population.size,
                'first_index': 0,
                'last_index': len(self.population),
                'first_id': int(self.population.first_id),
                'last_id': int(self.population.last_id),
                'label': self.population.label,
            }
        metadata.update(self.population.annotations)
        metadata['dt'] = self._simulator.state.dt # note that this has to run on all nodes (at least for NEST)
        return metadata

    def count(self, variable, gather=True, filter_ids=None):
        """
        Return the number of data points for each cell, as a dict. This is mainly
        useful for spike counts or for variable-time-step integration methods.
        """
        if variable == 'spikes':
            N = self._local_count(variable, filter_ids)
        else:
            raise Exception("Only implemented for spikes.")
        if gather and self._simulator.state.num_processes > 1:
            N = gather_dict(N)
        return N

    def store_to_cache(self, annotations={}):
        #make sure we haven't called get with clear=True since last reset
        #and that we did not do two resets in a row
        if (self._simulator.state.t != 0) and (not self.clear_flag):
            segment = self._get_current_segment()
            segment.annotate(**annotations)
            self.cache.store(segment)
        self.clear_flag = False
