"""
Defines classes and functions for managing recordings (spikes, membrane
potential etc).

These classes and functions are not part of the PyNN API, and are only for
internal use.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import numpy
import os
from copy import copy
from collections import defaultdict
from pyNN import errors
import neo
from datetime import datetime
import quantities as pq
try:
    basestring
except NameError:
    basestring = str

logger = logging.getLogger("PyNN")

MPI_ROOT = 0


def get_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        raise Exception("Trying to gather data without MPI installed. If you are not running a distributed simulation, this is a bug in PyNN.")
    return MPI.COMM_WORLD, {'DOUBLE': MPI.DOUBLE, 'SUM': MPI.SUM}

def rename_existing(filename):
    if os.path.exists(filename):
        os.system('mv %s %s_old' % (filename, filename))
        logger.warning("File %s already exists. Renaming the original file to %s_old" % (filename, filename))

def gather_array(data):
    # gather 1D or 2D numpy arrays
    mpi_comm, mpi_flags = get_mpi_comm()
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size = data.size
    sizes = mpi_comm.gather(size, root=MPI_ROOT) or []
    # now we pass the data
    displacements = [sum(sizes[:i]) for i in range(len(sizes))]
    gdata = numpy.empty(sum(sizes))
    mpi_comm.Gatherv([data.flatten(), size, mpi_flags['DOUBLE']],
                     [gdata, (sizes, displacements), mpi_flags['DOUBLE']],
                     root=MPI_ROOT)
    if len(data.shape) == 1:
        return gdata
    else:
        num_columns = data.shape[1]
        return gdata.reshape((gdata.size/num_columns, num_columns))



def gather_dict(D, all=False):
    # Note that if the same key exists on multiple nodes, the value from the
    # node with the highest rank will appear in the final dict.
    mpi_comm, mpi_flags = get_mpi_comm()
    if all:
        Ds = mpi_comm.allgather(D)
    else:
        Ds = mpi_comm.gather(D, root=MPI_ROOT)
    if Ds:
        for otherD in Ds:
            D.update(otherD)
    return D


def gather_blocks(data, ordered=True):
    """Gather Neo Blocks"""
    mpi_comm, mpi_flags = get_mpi_comm()
    assert isinstance(data, neo.Block)
    # for now, use gather_dict, which will probably be slow. Can optimize later
    D = {mpi_comm.rank: data}
    D = gather_dict(D)
    blocks = D.values()
    merged = data
    if mpi_comm.rank == MPI_ROOT:    
        merged = blocks[0]
        for block in blocks[1:]:
            merged.merge(block)
    if ordered:
        for segment in merged.segments:
            ordered_spiketrains = sorted(segment.spiketrains, key=lambda s: s.annotations['source_id'])
            segment.spiketrains = ordered_spiketrains
    return merged


def mpi_sum(x):
    mpi_comm, mpi_flags = get_mpi_comm()
    if mpi_comm.size > 1:
        return mpi_comm.allreduce(x, op=mpi_flags['SUM'])
    else:
        return x


def normalize_variables_arg(variables):
    """If variables is a single string, encapsulate it in a list."""
    if isinstance(variables, basestring) and variables != 'all':
        return [variables]
    else:
        return variables


def safe_makedirs(dir):
    """
    Version of makedirs not subject to race condition when using MPI.
    """
    if dir and not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != 17:
                raise


def get_io(filename):
    """
    Return a Neo IO instance, guessing the type based on the filename suffix.
    """
    logger.debug("Creating Neo IO for filename %s" % filename)
    dir = os.path.dirname(filename)
    safe_makedirs(dir)
    extension = os.path.splitext(filename)[1]
    if extension in ('.txt', '.ras', '.v', '.gsyn'):
        raise IOError("ASCII-based formats are not currently supported for output data. Try using the file extension '.pkl' or '.h5'")
    elif extension in ('.h5',):
        return neo.io.NeoHdf5IO(filename=filename)
    elif extension in ('.pkl', '.pickle'):
        return neo.io.PickleIO(filename=filename)
    elif extension == '.mat':
        return neo.io.NeoMatlabIO(filename=filename)
    else:  # function to be improved later
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


def remove_duplicate_spiketrains(data):
    for segment in data.segments:
        spiketrains = {}
        for spiketrain in segment.spiketrains:
            index = spiketrain.annotations["source_index"]
            spiketrains[index] = spiketrain
        min_index = min(spiketrains.keys())
        max_index = max(spiketrains.keys())
        segment.spiketrains = [spiketrains[i] for i in range(min_index, max_index+1)]
    return data


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
        self.population = population  # needed for writing header information
        self.recorded = defaultdict(set)
        self.cache = DataCache()
        self._simulator.state.recorders.add(self)
        self.clear_flag = False
        self._recording_start_time = self._simulator.state.t * pq.ms
        self.sampling_interval = self._simulator.state.dt

    def record(self, variables, ids, sampling_interval=None):
        """
        Add the cells in `ids` to the sets of recorded cells for the given variables.
        """
        logger.debug('Recorder.record(<%d cells>)' % len(ids))
        if sampling_interval is not None:
            if sampling_interval != self.sampling_interval and len(self.recorded) > 0:
                raise ValueError("All neurons in a population must be recorded with the same sampling interval.")

        ids = set([id for id in ids if id.local])
        for variable in normalize_variables_arg(variables):
            if not self.population.can_record(variable):
                raise errors.RecordingError(variable, self.population.celltype)
            new_ids = ids.difference(self.recorded[variable])
            self.recorded[variable] = self.recorded[variable].union(ids)
            self._record(variable, new_ids, sampling_interval)

    def reset(self):
        """Reset the list of things to be recorded."""
        self._reset()
        self.recorded = defaultdict(set)

    def filter_recorded(self, variable, filter_ids):
        if filter_ids is not None:
            return set(filter_ids).intersection(self.recorded[variable])
        else:
            return self.recorded[variable]

    def _get_current_segment(self, filter_ids=None, variables='all', clear=False):
        segment = neo.Segment(name="segment%03d" % self._simulator.state.segment_counter,
                              description=self.population.describe(),
                              rec_datetime=datetime.now()) # would be nice to get the time at the start of the recording, not the end
        variables_to_include = set(self.recorded.keys())
        if variables is not 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        for variable in variables_to_include:
            if variable == 'spikes':
                t_stop = self._simulator.state.t * pq.ms # must run on all MPI nodes
                segment.spiketrains = [
                    neo.SpikeTrain(self._get_spiketimes(id),
                                   t_start=self._recording_start_time,
                                   t_stop=t_stop,
                                   units='ms',
                                   source_population=self.population.label,
                                   source_id=int(id),
                                   source_index=self.population.id_to_index(id))
                    for id in sorted(self.filter_recorded('spikes', filter_ids))]
            else:
                ids = sorted(self.filter_recorded(variable, filter_ids))
                signal_array = self._get_all_signals(variable, ids, clear=clear)
                t_start = self._recording_start_time
                sampling_period = self.sampling_interval*pq.ms
                current_time = self._simulator.state.t*pq.ms
                mpi_node = self._simulator.state.mpi_rank  # for debugging
                if signal_array.size > 0:  # may be empty if none of the recorded cells are on this MPI node
                    channel_indices = numpy.array([self.population.id_to_index(id) for id in ids])
                    units = self.population.find_units(variable)
                    source_ids = numpy.fromiter(ids, dtype=int)
                    segment.analogsignalarrays.append(
                        neo.AnalogSignalArray(
                            signal_array,
                            units=units,
                            t_start=t_start,
                            sampling_period=sampling_period,
                            name=variable,
                            source_population=self.population.label,
                            channel_index=channel_indices,
                            source_ids=source_ids)
                    )
                    logger.debug("%d **** ids=%s, channels=%s", mpi_node, source_ids, channel_indices)
                    assert segment.analogsignalarrays[0].t_stop - current_time - 2*sampling_period < 1e-10
                    # need to add `Unit` and `RecordingChannelGroup` objects
        return segment

    def get(self, variables, gather=False, filter_ids=None, clear=False,
            annotations=None):
        """Return the recorded data as a Neo `Block`."""
        variables = normalize_variables_arg(variables)
        data = neo.Block()
        data.segments = [filter_by_variables(segment, variables)
                         for segment in self.cache]
        if self._simulator.state.running: # reset() has not been called, so current segment is not in cache
            data.segments.append(self._get_current_segment(filter_ids=filter_ids, variables=variables, clear=clear))
        data.name = self.population.label
        data.description = self.population.describe()
        data.rec_datetime = data.segments[0].rec_datetime
        data.annotate(**self.metadata)
        if annotations:
            data.annotate(**annotations)
        if gather and self._simulator.state.num_processes > 1:
            data = gather_blocks(data)
            if hasattr(self.population.celltype, "always_local") and self.population.celltype.always_local:
                data = remove_duplicate_spiketrains(data)
        if clear:
            self.clear()
        return data

    def clear(self):
        """
        Clear all recorded data, both from the cache and the simulator.
        """
        self.cache.clear()
        self.clear_flag = True
        self._recording_start_time = self._simulator.state.t * pq.ms
        self._clear_simulator()

    def write(self, variables, file=None, gather=False, filter_ids=None,
              clear=False, annotations=None):
        """Write recorded data to a Neo IO"""
        if isinstance(file, basestring):
            file = get_io(file)
        io = file or self.file
        if gather==False and self._simulator.state.num_processes > 1:
            io.filename += '.%d' % self._simulator.state.mpi_rank
        logger.debug("Recorder is writing '%s' to file '%s' with gather=%s" % (
                                               variables, io.filename, gather))
        data = self.get(variables, gather, filter_ids, clear, annotations=annotations)
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
                'simulator': self._simulator.name,
            }
        metadata.update(self.population.annotations)
        metadata['dt'] = self._simulator.state.dt # note that this has to run on all nodes (at least for NEST)
        metadata['mpi_processes'] = self._simulator.state.num_processes
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
        # make sure we haven't called get with clear=True since last reset
        # and that we did not do two resets in a row
        if (self._simulator.state.t != 0) and (not self.clear_flag):
            segment = self._get_current_segment()
            segment.annotate(**annotations)
            self.cache.store(segment)
        self.clear_flag = False
        self._recording_start_time = 0.0 * pq.ms
