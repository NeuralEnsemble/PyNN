# -*- coding: utf-8 -*-
"""
NEST v3 implementation of the PyNN API.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections import defaultdict
import logging
import numpy as np
import nest
from .. import recording, errors
from . import simulator

logger = logging.getLogger("PyNN")


def _set_status(obj, parameters):
    """Wrapper around nest.SetStatus() to add a more informative error message."""
    try:
        nest.SetStatus(obj, parameters)
    except nest.NESTError as e:
        raise nest.NESTError("%s. Parameter dictionary was: %s" % (e, parameters))


class RecordingDevice(object):
    """Base class for SpikeDetector and Multimeter"""

    def __init__(self, device_parameters, to_memory=True):
        # to be called at the end of the subclass __init__
        if to_memory:
            self.device.record_to = "memory"
        else:
            self.device.record_to = "ascii"
        self._all_ids = set([])
        self._connected = False
        self._overrun_data = None
        self._clean = True  # might there be data in the pipeline that hasn't been delivered yet
        simulator.state.recording_devices.append(self)
        _set_status(self.device, device_parameters)

    def add_ids(self, new_ids):
        assert not self._connected
        self._all_ids = self._all_ids.union(new_ids)

    def _get_data_arrays(self, variable, nest_variable, scale_factor, clear=False):
        """
        Return recorded data as pair of NumPy arrays: ids and values.
        """
        events = nest.GetStatus(self.device, 'events')[0]
        ids = events['senders']
        times = events["times"] - simulator.state._time_offset
        if variable == "times":
            values = times
        else:
            # I'm hoping numpy optimises for the case where scale_factor = 1,
            # otherwise should avoid this multiplication in that case
            values = events[nest_variable] * scale_factor

        valid_times_index = times <= simulator.state.t
        if clear:
            future_times_index = np.invert(valid_times_index)
            if future_times_index.any():
                new_overrun_data = {
                    "ids": ids[future_times_index],
                    "values": values[future_times_index]
                }
            else:
                new_overrun_data = None
            if self._overrun_data:
                ids = np.hstack((self._overrun_data["ids"], ids))
                values = np.hstack((self._overrun_data["values"], values))
            self._overrun_data = new_overrun_data
        else:
            ids = ids[valid_times_index]
            values = values[valid_times_index]
        return ids, values

    def get_data(self, variable, nest_variable, scale_factor, desired_ids, clear=False):
        """
        Return recorded data as a dictionary containing one numpy array for
        each neuron, ids as keys.
        """
        ids, values = self._get_data_arrays(variable, nest_variable, scale_factor, clear=clear)

        data = {}
        recorded_ids = set(ids)

        for id in recorded_ids:
            data[id] = []

        for id, v in zip(ids, values):
            data[id].append(v)

        desired_and_existing_ids = np.intersect1d(
            np.array(list(recorded_ids)), np.array(desired_ids))
        data = {k: data[k] for k in desired_and_existing_ids}

        if variable != 'times':
            if variable not in self._initial_values:
                self._initial_values[variable] = {}
            for id in desired_ids:
                initial_value = self._initial_values[variable].get(int(id),
                                                                   id.get_initial_value(variable))
                if self._clean:
                    # NEST does not record values at the zeroth time step, so we
                    # add them here.
                    data[int(id)] = [initial_value] + data.get(int(id), [])
                else:
                    # The values at the zeroth time step come from a previous run,
                    # so should be replaced
                    if len(data[int(id)]) > 0:
                        data[int(id)][0] = initial_value

                # if `get_data(..., clear=True)` is called in the middle of a simulation, the
                # value at the last time point will become the initial value for
                # the next time `get_data()` is called
                if clear:
                    self._initial_values[variable][int(id)] = data[int(id)][-1]

        return data


class SpikeDetector(RecordingDevice):
    """A wrapper around the NEST spike_recorder device"""

    def __init__(self, to_memory=True):
        self.device = nest.Create('spike_recorder')
        device_parameters = {}
        if not to_memory:
            device_parameters["precision"] = simulator.state.default_recording_precision
        super(SpikeDetector, self).__init__(device_parameters, to_memory)

    def connect_to_cells(self):
        assert not self._connected
        if len(self._all_ids) > 0:
            nest.Connect(nest.NodeCollection(sorted(self._all_ids)),
                         self.device,
                         {'rule': 'all_to_all'},
                         {'delay': simulator.state.min_delay})
        self._connected = True

    def get_spiketimes(self, desired_ids, clear=False):
        """
        Return spike times as a dictionary containing one numpy array for
        each neuron, ids as keys.

        Equivalent to `get_data('times', desired_ids)`
        """
        id_array, times_array = self._get_data_arrays("times", "times", 1, clear=clear)
        recorded_ids = np.unique(id_array)
        desired_and_existing_ids = np.intersect1d(recorded_ids, np.array(desired_ids))
        mask = np.in1d(id_array, desired_and_existing_ids)
        return id_array[mask], times_array[mask]

    def get_spike_counts(self, desired_ids):
        events = nest.GetStatus(self.device, 'events')[0]
        N = {}
        for id in desired_ids:
            mask = events['senders'] == int(id)
            N[int(id)] = len(events['times'][mask])
        return N


class Multimeter(RecordingDevice):
    """A wrapper around the NEST multimeter device"""

    def __init__(self, to_memory=True):
        self.device = nest.Create('multimeter')
        device_parameters = {
            "interval": simulator.state.dt,
        }
        self._initial_values = {}
        super(Multimeter, self).__init__(device_parameters, to_memory)

    def connect_to_cells(self):
        assert not self._connected
        if len(self._all_ids) > 0:
            nest.Connect(self.device,
                         nest.NodeCollection(sorted(self._all_ids)),
                         {'rule': 'all_to_all'},
                         {'delay': simulator.state.min_delay})
        self._connected = True

    @property
    def variables(self):
        return set(nest.GetStatus(self.device, 'record_from')[0])

    def add_variable(self, variable):
        """variable should be the NEST variable name"""
        current_variables = self.variables
        current_variables.add(variable)
        _set_status(self.device, {'record_from': list(current_variables)})


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
    scale_factors = {'spikes': 1,
                     'v': 1,
                     'w': 0.001,
                     'gsyn': 0.001}  # units conversion

    def __init__(self, population, file=None):
        self._multimeter = Multimeter()
        self._spike_detector = SpikeDetector()
        recording.Recorder.__init__(self, population, file)
        self.recorded_all = defaultdict(set)

    def record(self, variables, ids, sampling_interval=None):
        """
        Add the cells in `ids` to the sets of recorded cells for the given variables.
        """
        logger.debug('Recorder.record(<%d cells>)' % len(ids))
        self._check_sampling_interval(sampling_interval)

        # for NEST we need all ids, not just local ones, otherwise simulations
        # sometimes hang with MPI if some nodes aren't recording anything
        all_ids = set(ids)
        local_ids = set([id for id in ids if id.local])
        for variable in recording.normalize_variables_arg(variables):
            if not self.population.can_record(variable):
                raise errors.RecordingError(variable, self.population.celltype)
            new_ids = all_ids.difference(self.recorded_all[variable])
            self.recorded[variable] = self.recorded[variable].union(local_ids)
            self.recorded_all[variable] = self.recorded_all[variable].union(all_ids)
            self._record(variable, new_ids, sampling_interval)

    def _record(self, variable, new_ids, sampling_interval=None):
        """
        Add the cells in `new_ids` to the set of recorded cells for the given
        variable. Since a given node can only be recorded from by one multimeter
        (http://www.nest-initiative.org/index.php/Analog_recording_with_multimeter, 14/11/11)
        we record all analog variables for all requested cells.
        """
        if variable == 'spikes':
            self._spike_detector.add_ids(new_ids)
        else:
            self.sampling_interval = sampling_interval
            if hasattr(self.population.celltype, "variable_map"):
                # only true for PyNN standard cells
                nest_variable = self.population.celltype.variable_map[variable]
            else:
                nest_variable = variable
            self._multimeter.add_variable(nest_variable)
            self._multimeter.add_ids(new_ids)

    def _get_sampling_interval(self):
        return nest.GetStatus(self._multimeter.device, "interval")[0]

    def _set_sampling_interval(self, value):
        if value is not None:
            nest.SetStatus(self._multimeter.device, {"interval": value})
    sampling_interval = property(fget=_get_sampling_interval,
                                 fset=_set_sampling_interval)

    def _reset(self):
        """ """
        simulator.state.recording_devices.remove(self._multimeter)
        simulator.state.recording_devices.remove(self._spike_detector)
        # I guess the existing devices still exist in NEST, can we delete them
        # or at least turn them off?
        # Maybe we can reset them, rather than create new ones?
        self._multimeter = Multimeter()
        self._spike_detector = SpikeDetector()

    def _get_spiketimes(self, ids, clear=False):
        return self._spike_detector.get_spiketimes(ids, clear=clear)

    def _get_all_signals(self, variable, ids, clear=False):
        if hasattr(self.population.celltype, "variable_map"):
            nest_variable = self.population.celltype.variable_map[variable]
        else:
            nest_variable = variable
        if hasattr(self.population.celltype, "scale_factors"):
            scale_factor = self.population.celltype.scale_factors[variable]
        else:
            scale_factor = 1
        data = self._multimeter.get_data(variable, nest_variable, scale_factor, ids, clear=clear)
        times = None
        if len(ids) > 0:
            # JACOMMENT: this is very expensive but not sure how to get rid of it
            return np.array([data[i] for i in ids]).T, times
        else:
            return np.array([]), times

    def _local_count(self, variable, filter_ids):
        assert variable == 'spikes'
        return self._spike_detector.get_spike_counts(self.filter_recorded('spikes', filter_ids))

    def _clear_simulator(self):
        """
        Should remove all recorded data held by the simulator and, ideally,
        free up the memory.
        """
        for rec in (self._spike_detector, self._multimeter):
            nest.SetStatus(rec.device, 'n_events', 0)
            rec._clean = False

    def store_to_cache(self, annotations=None):
        # we over-ride the implementation from the parent class so as to
        # do some reinitialisation.
        recording.Recorder.store_to_cache(self, annotations)
        self._multimeter._initial_values = {}
