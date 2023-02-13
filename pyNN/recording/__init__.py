"""
Defines classes and functions for managing recordings (spikes, membrane
potential etc).

These classes and functions are not part of the PyNN API, and are only for
internal use.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
from datetime import datetime
import os
from copy import copy
from collections import defaultdict
from warnings import warn

import numpy as np
import neo
import quantities as pq

from .. import errors


logger = logging.getLogger("PyNN")

MPI_ROOT = 0


def get_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        raise Exception(
            "Trying to gather data without MPI installed. "
            "If you are not running a distributed simulation, this is a bug in PyNN.")
    return MPI.COMM_WORLD, {'DOUBLE': MPI.DOUBLE, 'SUM': MPI.SUM}


def rename_existing(filename):
    if os.path.exists(filename):
        os.system('mv %s %s_old' % (filename, filename))
        logger.warning(f"File {filename} already exists. "
                       "Renaming the original file to {filename}_old")


def gather_array(data):
    # gather 1D or 2D numpy arrays
    mpi_comm, mpi_flags = get_mpi_comm()
    assert isinstance(data, np.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size = data.size
    sizes = mpi_comm.gather(size, root=MPI_ROOT) or []
    # now we pass the data
    displacements = [sum(sizes[:i]) for i in range(len(sizes))]
    gdata = np.empty(sum(sizes))
    mpi_comm.Gatherv([data.flatten(), size, mpi_flags['DOUBLE']],
                     [gdata, (sizes, displacements), mpi_flags['DOUBLE']],
                     root=MPI_ROOT)
    if len(data.shape) == 1:
        return gdata
    else:
        num_columns = data.shape[1]
        return gdata.reshape((gdata.size / num_columns, num_columns))


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
    blocks = list(D.values())
    merged = data
    if mpi_comm.rank == MPI_ROOT:
        merged = blocks[0]
        # the following business with setting sig.segment is a workaround for a bug in Neo
        for seg in merged.segments:
            for sig in seg.analogsignals:
                sig.segment = seg
        for block in blocks[1:]:
            for seg, mseg in zip(block.segments, merged.segments):
                for sig in seg.analogsignals:
                    sig.segment = mseg
            merged.merge(block)
    if ordered:
        for segment in merged.segments:
            ordered_spiketrains = sorted(
                segment.spiketrains, key=lambda s: s.annotations['source_id'])
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
    if isinstance(variables, str) and variables != 'all':
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
        raise IOError(
            "ASCII-based formats are not currently supported for output data. "
            "Try using the file extension '.nwb', '.nix', '.pkl' or '.h5'")
    elif extension in ('.nix', '.h5'):
        return neo.io.NixIO(filename=filename)
    elif extension in ('.nwb',):
        return neo.io.NWBIO(filename=filename, mode="w")
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
        new_segment = copy(segment)  # shallow copy
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
        segment.spiketrains = [spiketrains[i] for i in range(min_index, max_index + 1)]
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
        if hasattr(self._simulator.state, "record_sample_times"):
            self.record_times = self._simulator.state.record_sample_times
        else:
            self.record_times = False

    def record(self, variables, ids, sampling_interval=None):
        """
        Add the cells in `ids` to the sets of recorded cells for the given variables.
        """
        logger.debug('Recorder.record(<%d cells>)' % len(ids))
        self._check_sampling_interval(sampling_interval)

        ids = set([id for id in ids if id.local])
        for variable in normalize_variables_arg(variables):
            if not self.population.can_record(variable):
                raise errors.RecordingError(variable, self.population.celltype)
            new_ids = ids.difference(self.recorded[variable])
            self.recorded[variable] = self.recorded[variable].union(ids)
            self._record(variable, new_ids, sampling_interval)

    def _check_sampling_interval(self, sampling_interval):
        """
        Check whether record() has been called previously with a different sampling interval
        (we exclude recording of spikes, as the sampling interval does not apply in that case)
        """
        if sampling_interval is not None and sampling_interval != self.sampling_interval:
            recorded_variables = list(self.recorded.keys())
            if "spikes" in recorded_variables:
                recorded_variables.remove("spikes")
            if len(recorded_variables) > 0:
                raise ValueError(
                    "All neurons in a population must be recorded "
                    "with the same sampling interval.")

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
                              # would be nice to get the time at the start of the recording,
                              # not the end
                              rec_datetime=datetime.now())
        variables_to_include = set(self.recorded.keys())
        if variables != 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        for variable in variables_to_include:
            if variable == 'spikes':
                t_stop = self._simulator.state.t * pq.ms  # must run on all MPI nodes
                sids = sorted(self.filter_recorded('spikes', filter_ids))
                data = self._get_spiketimes(sids, clear=clear)

                if isinstance(data, dict):
                    for id in sids:
                        times = pq.Quantity(data.get(int(id), []), pq.ms)
                        if times.size > 0 and times.max() > t_stop:
                            warn("Recorded at least one spike after t_stop")
                            times = times[times <= t_stop]
                        segment.spiketrains.append(
                            neo.SpikeTrain(
                                times,
                                t_start=self._recording_start_time,
                                t_stop=t_stop,
                                units='ms',
                                source_population=self.population.label,
                                source_id=int(id),
                                source_index=self.population.id_to_index(int(id)))
                        )
                        for train in segment.spiketrains:
                            train.segment = segment
                else:
                    assert isinstance(data, tuple)
                    id_array, times = data
                    times *= pq.ms
                    if times.size > 0 and times.max() > t_stop:
                        warn("Recorded at least one spike after t_stop")
                        mask = times <= t_stop
                        times = times[mask]
                        id_array = id_array[mask]
                    segment.spiketrains = neo.spiketrainlist.SpikeTrainList.from_spike_time_array(
                        times, id_array,
                        np.array(sids, dtype=int),
                        t_stop=t_stop,
                        units="ms",
                        t_start=self._recording_start_time,
                        source_population=self.population.label
                    )
                    segment.spiketrains.segment = segment
            else:
                ids = sorted(self.filter_recorded(variable, filter_ids))
                signal_array, times_array = self._get_all_signals(variable, ids, clear=clear)
                mpi_node = self._simulator.state.mpi_rank  # for debugging
                if signal_array.size > 0:
                    # may be empty if none of the recorded cells are on this MPI node
                    units = self.population.find_units(variable)
                    source_ids = np.fromiter(ids, dtype=int)
                    channel_index = np.array([self.population.id_to_index(id) for id in ids])
                    if self.record_times:
                        if signal_array.shape == times_array.shape:
                            # in the current version of Neo, all channels in
                            # IrregularlySampledSignal must have the same sample times,
                            # so we need to create here a list of signals
                            signals = [
                                neo.IrregularlySampledSignal(
                                    times_array[:, i],
                                    signal_array[:, i],
                                    units=units,
                                    time_units=pq.ms,
                                    name=variable,
                                    source_ids=[source_id],
                                    source_population=self.population.label,
                                    array_annotations={"channel_index": [i]}
                                )
                                for i, source_id in zip(channel_index, source_ids)
                            ]
                        else:
                            # all channels have the same sample times
                            assert signal_array.shape[0] == times_array.size
                            signals = [
                                neo.IrregularlySampledSignal(
                                    times_array, signal_array, units=units, time_units=pq.ms,
                                    name=variable, source_ids=source_ids,
                                    source_population=self.population.label,
                                    array_annotations={"channel_index": channel_index}
                                )
                            ]
                        segment.irregularlysampledsignals.extend(signals)
                        for signal in signals:
                            signal.segment = segment
                    else:
                        t_start = self._recording_start_time
                        t_stop = self._simulator.state.t * pq.ms
                        sampling_period = self.sampling_interval * pq.ms
                        current_time = self._simulator.state.t * pq.ms
                        signal = neo.AnalogSignal(
                            signal_array,
                            units=units,
                            t_start=t_start,
                            sampling_period=sampling_period,
                            name=variable, source_ids=source_ids,
                            source_population=self.population.label,
                            array_annotations={"channel_index": channel_index}
                        )
                        assert signal.t_stop - current_time - 2 * sampling_period < 1e-10
                        logger.debug("%d **** ids=%s, channels=%s", mpi_node,
                                     source_ids, signal.array_annotations["channel_index"])
                        segment.analogsignals.append(signal)
                        signal.segment = segment
        return segment

    def get(self, variables, gather=False, filter_ids=None, clear=False,
            annotations=None):
        """Return the recorded data as a Neo `Block`."""
        variables = normalize_variables_arg(variables)
        data = neo.Block()
        data.segments = [filter_by_variables(segment, variables)
                         for segment in self.cache]
        if self._simulator.state.running:
            # reset() has not been called, so current segment is not in cache
            data.segments.append(self._get_current_segment(
                filter_ids=filter_ids, variables=variables, clear=clear))
        for segment in data.segments:
            segment.block = data
        data.name = self.population.label
        data.description = self.population.describe()
        data.rec_datetime = data.segments[0].rec_datetime
        data.annotate(**self.metadata)
        if annotations:
            data.annotate(**annotations)
        if gather and self._simulator.state.num_processes > 1:
            data = gather_blocks(data)
            if (
                hasattr(self.population.celltype, "always_local")
                and self.population.celltype.always_local
            ):
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
        if isinstance(file, str):
            file = get_io(file)
        io = file or self.file
        if gather is False and self._simulator.state.num_processes > 1:
            io.filename += '.%d' % self._simulator.state.mpi_rank
        logger.debug("Recorder is writing '%s' to file '%s' with gather=%s" % (
            variables, io.filename, gather))
        data = self.get(variables, gather, filter_ids, clear, annotations=annotations)
        if self._simulator.state.mpi_rank == 0 or gather is False:
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
        # note that this has to run on all nodes (at least for NEST)
        metadata['dt'] = self._simulator.state.dt
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

    def store_to_cache(self, annotations=None):
        # make sure we haven't called get with clear=True since last reset
        # and that we did not do two resets in a row
        if (self._simulator.state.t != 0) and (not self.clear_flag):
            if annotations is None:
                annotations = {}
            segment = self._get_current_segment()
            segment.annotate(**annotations)
            self.cache.store(segment)
        self.clear_flag = False
        self._recording_start_time = 0.0 * pq.ms
