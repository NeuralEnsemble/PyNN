"""

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections import defaultdict
import logging
import numpy as np
from .. import recording
from .. import errors
from ..morphology import LocationGenerator
from .morphology import Location, LabelledLocations
from . import simulator
import re
from neuron import h


logger = logging.getLogger("PyNN")


recordable_pattern = re.compile(
    r'((?P<section>\w+)(\((?P<location>[-+]?[0-9]*\.?[0-9]+)\))?\.)?(?P<var>\w+)')


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator


    def record(self, variables, ids, sampling_interval=None, locations=None):
        """
        Add the cells in `ids` to the sets of recorded cells for the given variables.
        """
        logger.debug('Recorder.record(<%d cells>)' % len(ids))
        self._check_sampling_interval(sampling_interval)

        if isinstance(variables, str) and variables != 'all':
            variables = [variables]

        ids = set([id for id in ids if id.local])

        if locations is None:  # point neurons
            for var_path in variables:
                if not self.population.can_record(var_path, None):
                    raise errors.RecordingError(var_path, self.population.celltype)
                var_obj = recording.Variable(location=None, name=var_path, label=None)
                new_ids = ids.difference(self.recorded[var_obj])
                self.recorded[var_obj] = self.recorded[var_obj].union(ids)
                self._record(var_obj, new_ids, sampling_interval)

        else:  # multi-compartment neurons
            if not isinstance(locations, (list, tuple)):
                assert isinstance(locations, (str, LocationGenerator))
                locations = [locations]

            resolved_variables = defaultdict(set)
            for item in locations:
                if isinstance(item, str):
                    location_generator = LabelledLocations(item)
                elif isinstance(item, LocationGenerator):
                    location_generator = item
                else:
                    raise ValueError("'locations' should be a str, list, LocationGenerator or None")

                # todo: avoid this loop if all the cells in the population have an identical morphology
                for id in ids:
                    morphology = id._cell.morphology
                    # in principle, generate_locations() could give different locations for
                    # cells with different morphologies, so we construct a dict containing
                    # the ids of the neurons for which a given location exists
                    for location in location_generator.generate_locations(morphology, label_prefix="", cell=id._cell):
                        for var_name in variables:
                            var_obj = recording.Variable(location=location, name=var_name, label=location)  # better labels? include section id?
                            resolved_variables[var_obj].add(id)

            for var_obj, id_list in resolved_variables.items():
                new_ids = id_list.difference(self.recorded[var_obj])
                self.recorded[var_obj] = self.recorded[var_obj].union(id_list)
                self._record(var_obj, new_ids, sampling_interval)

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if variable.name == 'spikes':
            for id in new_ids:
                if id._cell.rec is not None:
                    id._cell.rec.record(id._cell.spike_times)
                else:  # SpikeSourceArray
                    id._cell.recording = True
        else:
            self.sampling_interval = sampling_interval or self._simulator.state.dt
            for id in new_ids:
                self._record_state_variable(id._cell, variable)

    def _record_state_variable(self, cell, variable):
        if variable.location is None:
            if hasattr(cell, 'recordable') and variable in cell.recordable:
                hoc_var = cell.recordable[variable]
            elif variable.name == 'v':
                hoc_var = cell.source_section(0.5)._ref_v  # or use "seg.v"?
            elif variable.name == 'gsyn_exc':
                hoc_var = cell.esyn._ref_g
            elif variable.name == 'gsyn_inh':
                hoc_var = cell.isyn._ref_g
            else:
                source, var_name = self._resolve_variable(cell, variable.name)
                if hasattr(self.population.celltype, "variable_map"):
                    var_name = self.population.celltype.variable_map[variable.name]
                hoc_var = getattr(source, "_ref_%s" % var_name)
            hoc_vars = [hoc_var]
        else:
            assert isinstance(variable.location, str)
            hoc_vars = []

            cell_location = cell.locations[variable.location]
            source = cell_location.section(cell_location.position)
            if variable.name == 'v':
                hoc_vars.append(source._ref_v)
            else:
                ion_channel, var_name = variable.name.split(".")
                mechanism_name, hoc_var_name = self.population.celltype.ion_channels[ion_channel].variable_translations[var_name]
                mechanism = getattr(source, mechanism_name)
                hoc_vars.append(getattr(mechanism, "_ref_{}".format(hoc_var_name)))
        for hoc_var in hoc_vars:
            vec = h.Vector()
            if self.sampling_interval == self._simulator.state.dt or self.record_times:
                vec.record(hoc_var)
            else:
                vec.record(hoc_var, self.sampling_interval)
            cell.traces[variable].append(vec)
        if not cell.recording_time:
            cell.recorded_times = h.Vector()
            if self.sampling_interval == self._simulator.state.dt or self.record_times:
                cell.recorded_times.record(h._ref_t)
            else:
                cell.recorded_times.record(h._ref_t, self.sampling_interval)
            cell.recording_time += 1

    # could be staticmethod
    def _resolve_variable(self, cell, variable_path):
        match = recordable_pattern.match(variable_path)
        if match:
            parts = match.groupdict()
            if parts['section']:
                section = getattr(cell, parts['section'])
                if parts['location']:
                    source = section(float(parts['location']))
                else:
                    source = section
            else:
                source = cell.source
            return source, parts['var']
        else:
            raise AttributeError("Recording of %s not implemented." % variable_path)

    def _reset(self):
        """Reset the list of things to be recorded."""
        for id in set.union(*self.recorded.values()):
            id._cell.traces = defaultdict(list)
            id._cell.spike_times = h.Vector(0)
        id._cell.recording_time == 0
        id._cell.recorded_times = None

    def _clear_simulator(self):
        """
        Should remove all recorded data held by the simulator and, ideally,
        free up the memory.
        """
        for id in set.union(*self.recorded.values()):
            if hasattr(id._cell, "traces"):
                for variable in id._cell.traces:
                    for vec in id._cell.traces[variable]:
                        vec.resize(0)
            if id._cell.rec is not None:
                id._cell.spike_times.resize(0)
            else:
                id._cell.clear_past_spikes()

    def _get_spiketimes(self, id, clear=False):
        if hasattr(id, "__len__"):
            all_spiketimes = {}
            for cell_id in id:
                if cell_id._cell.rec is None:  # SpikeSourceArray
                    spikes = cell_id._cell.get_recorded_spike_times()
                else:
                    spikes = cell_id._cell.spike_times.as_numpy()
                all_spiketimes[cell_id] = spikes[spikes <= simulator.state.t + 1e-9]
            return all_spiketimes
        else:
            spikes = id._cell.spike_times.as_numpy()
            return spikes[spikes <= simulator.state.t + 1e-9]

    def _get_all_signals(self, variable, ids, clear=False):
        times = None
        if len(ids) > 0:
            # note: id._cell.traces[variable] is a list of Vectors, one per segment
            signals = np.vstack([vec for id in ids for vec in id._cell.traces[variable]]).T
            if self.record_times:
                assert not simulator.state.cvode.use_local_dt()
                # the following line assumes all cells are sampled at the same time
                # which should be true if cvode.use_local_dt() returns False
                times = np.array(ids[0]._cell.recorded_times)
            else:
                expected_length = np.rint(simulator.state.tstop / self.sampling_interval) + 1
                if signals.shape[0] != expected_length:
                    # generally due to floating point/rounding issues
                    signals = np.vstack((signals, signals[-1, :]))
                if ".isyn" in variable:
                    # this is a hack, since negative currents in NMODL files
                    # correspond to positive currents in PyNN
                    # todo: reimplement this in a more robust way
                    signals *= -1
        else:
            signals = np.array([])
        return signals, times

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable.name == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = id._cell.spike_times.size()
        else:
            raise Exception("Only implemented for spikes")
        return N
