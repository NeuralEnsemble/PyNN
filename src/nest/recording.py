"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import tempfile
import os
import numpy
import logging
import warnings
import nest
from pyNN import recording, errors
from pyNN.nest import simulator
import neo
from datetime import datetime
import quantities as pq

VARIABLE_MAP = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in'}
REVERSE_VARIABLE_MAP = dict((v,k) for k,v in VARIABLE_MAP.items())

logger = logging.getLogger("PyNN")


def _set_status(obj, parameters):
    """Wrapper around nest.SetStatus() to add a more informative error message."""
    try:
        nest.SetStatus(obj, parameters)
    except nest.hl_api.NESTError, e:
        raise nest.hl_api.NESTError("%s. Parameter dictionary was: %s" % (e, parameters))


class RecordingDevice(object):
    """Base class for SpikeDetector and Multimeter"""
    
    def __init__(self, device_parameters, to_memory=True):
        # to be called at the end of the subclass __init__
        device_parameters.update(withgid=True, withtime=True)
        if to_memory:
            device_parameters.update(to_file=False, to_memory=True)
        else:
            device_parameters.update(to_file=True, to_memory=False)
        self._all_ids = set([])
        self._connected = False
        simulator.recording_devices.append(self)
        _set_status(self.device, device_parameters)

    def add_ids(self, new_ids):
        assert not self._connected
        self._all_ids = self._all_ids.union(new_ids)
    
    def get_data(self, variable, desired_ids):
        """
        Return recorded data as a dictionary containing one numpy array for
        each neuron, ids as keys.
        """
        events = nest.GetStatus(self.device,'events')[0]
        ids = events['senders']
        values = events[variable]
        data = {}
        for id in desired_ids:
            data[id] = values[ids==id]
        return data
    

class SpikeDetector(RecordingDevice):
    """A wrapper around the NEST spike_detector device"""
    _nest_connect = lambda device, ids: nest.ConvergentConnect()

    def __init__(self, to_memory=True):
        self.device = nest.Create('spike_detector')
        device_parameters = {      
            "precise_times": True,
            "precision": simulator.state.default_recording_precision
        }
        super(SpikeDetector, self).__init__(device_parameters, to_memory)

    def connect_to_cells(self):
        assert not self._connected
        nest.ConvergentConnect(list(self._all_ids), self.device, model='static_synapse')
        self._connected = True
    
    def get_spiketimes(self, desired_ids):
        """
        Return spike times as a dictionary containing one numpy array for
        each neuron, ids as keys.
        
        Equivalent to `get_data('times', desired_ids)`
        """
        return self.get_data('times', desired_ids)
        

class Multimeter(RecordingDevice):
    """A wrapper around the NEST multimeter device"""
    _nest_connect = nest.ConvergentConnect
    
    def __init__(self, to_memory=True):
        self.device = nest.Create('multimeter')
        device_parameters = {
            "interval": simulator.state.dt,
        }
        super(Multimeter, self).__init__(device_parameters, to_memory)

    def connect_to_cells(self):
        assert not self._connected
        nest.DivergentConnect(self.device, list(self._all_ids), model='static_synapse')
        self._connected = True
    
    @property
    def variables(self):
        return set(nest.GetStatus(self.device, 'record_from')[0])
    
    def add_variable(self, variable):
        current_variables = self.variables
        current_variables.add(VARIABLE_MAP.get(variable, variable))
        _set_status(self.device, {'record_from': list(current_variables)})
    



#class RecordingDevice(object):
#    scale_factors = {'V_m': 1, 'g_ex': 0.001, 'g_in': 0.001}
#    
#    def __init__(self, device_type, to_memory=False):
#        assert device_type in ("multimeter", "spike_detector")
#        self.type      = device_type
#        self.device    = nest.Create(device_type)
#        self.to_memory = to_memory
#        device_parameters = {"withgid": True, "withtime": True}
#        if self.type is 'multimeter':
#            device_parameters["interval"] = simulator.state.dt
#        else:
#            device_parameters["precise_times"] = True
#            device_parameters["precision"] = simulator.state.default_recording_precision
#        if to_memory:
#            device_parameters.update(to_file=False, to_memory=True)
#        else:
#            device_parameters.update(to_file=True, to_memory=False)
#        try:
#            nest.SetStatus(self.device, device_parameters)
#        except nest.hl_api.NESTError, e:
#            raise nest.hl_api.NESTError("%s. Parameter dictionary was: %s" % (e, device_parameters))
#        
#        self.record_from = []
#        self._local_files_merged = False
#        self._gathered = False
#        self._connected = False
#        self._all_ids = set([])
#        simulator.recording_devices.append(self)
#        logger.debug("Created %s with parameters %s" % (device_type, device_parameters))
#
#    def __del__(self):
#        for name in "_merged_file", "_gathered_file":
#            if hasattr(self, name):
#                getattr(self, name).close()
#
#    def add_variables(self, *variables):
#        assert self.type is "multimeter", "Can't add variables to a spike detector"
#        self.record_from.extend(variables)
#        nest.SetStatus(self.device, {'record_from': self.record_from})
#
#    def add_cells(self, new_ids):
#        self._all_ids = self._all_ids.union(new_ids)
#        
#    def connect_to_cells(self):
#        if not self._connected:
#            ids = list(self._all_ids)
#            if self.type is "spike_detector":
#                nest.ConvergentConnect(ids, self.device, model='static_synapse')
#            else:
#                nest.DivergentConnect(self.device, ids, model='static_synapse')
#            self._connected = True
#
#    def in_memory(self):
#        """Determine whether data is being recorded to memory."""
#        return nest.GetStatus(self.device, 'to_memory')[0]
#    
#    def events_to_array(self, events):
#        """
#        Transform the NEST events dictionary (when recording to memory) to a
#        Numpy array.
#        """
#        ids = events['senders']
#        times = events['times']
#        if self.type == 'spike_detector':
#            data = numpy.array((ids, times)).T
#        else:
#            data = [ids, times]
#            for var in self.record_from:
#                data.append(events[var])
#            data = numpy.array(data).T
#        return data
#
#    def scale_data(self, data):
#        """
#        Scale the data to give appropriate units.
#        """
#        scale_factors = [RecordingDevice.scale_factors.get(var, 1)
#                         for var in self.record_from]
#        for i, scale_factor in enumerate(scale_factors):
#            column = i+2 # first two columns are id and t, which should not be scaled.
#            if scale_factor != 1:
#                data[:, column] *= scale_factor 
#        return data
#
#    def add_initial_values(self, data):
#        """
#        Add initial values (NEST does not record the value at t=0).
#        """
#        logger.debug("Prepending initial values to recorded data")
#        initial_values = []
#        for id in self._all_ids:
#            initial = [id, 0.0]
#            for variable in self.record_from:
#                variable = REVERSE_VARIABLE_MAP.get(variable, variable)
#                try:
#                    initial.append(id.get_initial_value(variable))
#                except KeyError:
#                    initial.append(0.0) # unsatisfactory
#            initial_values.append(initial)    
#        if initial_values:
#            data = numpy.concatenate((initial_values, data))
#        return data
#
#    def read_data_from_memory(self, gather, compatible_output):
#        """
#        Return memory-recorded data.
#        
#        `gather` -- if True, gather data from all MPI nodes.
#        `compatible_output` -- if True, transform the data into the PyNN
#                               standard format.
#        """
#        data = nest.GetStatus(self.device,'events')[0]
#        if compatible_output:
#            data = self.events_to_array(data)
#            data = self.scale_data(data)  
#        if gather and simulator.state.num_processes > 1:
#            data = recording.gather(data)     
#            self._gathered_file = tempfile.TemporaryFile()
#            numpy.save(self._gathered_file, data)
#            self._gathered = True
#        return data
#    
#    def read_local_data(self, compatible_output):
#        """
#        Return file-recorded data from the local MPI node, merging data from
#        different threads if applicable.
#        
#        The merged data is cached, to avoid the overhead of re-merging if the
#        method is called again.
#        """
#        # what if the method is called with different values of
#        # `compatible_output`? Need to cache these separately.
#        if self._local_files_merged:
#            self._merged_file.seek(0)
#            data = numpy.load(self._merged_file)
#        else:
#            d = nest.GetStatus(self.device)[0]
#            if "filenames" in d:
#                nest_files = d['filenames']
#            else: # indicates that run() has not been called.
#                raise errors.NothingToWriteError("No recorder data. Have you called run()?")   
#            # possibly we can just keep on saving to the end of self._merged_file, instead of concatenating everything in memory
#            logger.debug("Concatenating data from the following files: %s" % ", ".join(nest_files))
#            non_empty_nest_files = [filename for filename in nest_files if os.stat(filename).st_size > 0]
#            if len(non_empty_nest_files) > 0:
#                data = numpy.concatenate([numpy.loadtxt(nest_file, dtype=float) for nest_file in non_empty_nest_files])
#            if len(non_empty_nest_files) == 0 or data.size == 0:
#                if self.type is "spike_detector":
#                    ncol = 2
#                else:
#                    ncol = 2 + len(self.record_from)
#                data = numpy.empty([0, ncol])
#            if compatible_output and self.type is not "spike_detector":
#                data = self.scale_data(data)
#                data = self.add_initial_values(data)
#            self._merged_file = tempfile.TemporaryFile()
#            numpy.save(self._merged_file, data)
#            self._local_files_merged = True
#        return data
#    
#    def read_data(self, gather, compatible_output, always_local=False):
#        """
#        Return file-recorded data.
#        
#        `gather` -- if True, gather data from all MPI nodes.
#        `compatible_output` -- if True, transform the data into the PyNN
#                               standard format.
#                               
#        Gathered data is cached, so the MPI communication need only be done
#        once, even if the method is called multiple times.
#        """
#        # what if the method is called with different values of
#        # `compatible_output`? Need to cache these separately.
#        if not self.to_memory:
#            if gather and simulator.state.num_processes > 1:
#                if self._gathered:
#                    logger.debug("Loading previously gathered data from cache")
#                    self._gathered_file.seek(0)
#                    data = numpy.load(self._gathered_file)
#                else:
#                    local_data = self.read_local_data(compatible_output)
#                    if always_local:
#                        data = local_data # for always_local cells, no need to gather
#                    else:
#                        logger.debug("Gathering data")
#                        data = recording.gather(local_data)
#                    logger.debug("Caching gathered data")
#                    self._gathered_file = tempfile.TemporaryFile()
#                    numpy.save(self._gathered_file, data)
#                    self._gathered = True
#            else:
#                data = self.read_local_data(compatible_output)
#            if len(data.shape) == 1:
#                data = data.reshape((1, data.size))
#            return data
#        else:
#            return self.read_data_from_memory(gather, compatible_output)
#    
#    def read_subset(self, variables, gather, compatible_output, always_local=False):
#        if self.in_memory():
#            data = self.read_data_from_memory(gather, compatible_output)
#        else: # in file
#            data = self.read_data(gather, compatible_output, always_local)
#        indices = []
#        for variable in variables:
#            try:
#                indices.append(self.record_from.index(variable))
#            except ValueError:
#                raise Exception("%s not recorded" % variable)
#        columns = tuple([0, 1] + [index + 2 for index in indices])
#        return data[:, columns]


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
    scale_factors = {'spikes': 1,
                     'v': 1,
                     'gsyn': 0.001} # units conversion
    
    def __init__(self, population, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, population, file)
        self._multimeter = Multimeter()
        self._spike_detector = SpikeDetector()
#        self._create_device()

#    def _create_device(self, variable):
#        to_memory = (self.file is False) # note file=None means we save to a temporary file, not to memory
#        if variable is "spikes":
#            self._device = RecordingDevice("spike_detector", to_memory)
#        else:
#            self._device = None
#            for recorder in self.population.recorders.values():
#                if hasattr(recorder, "_device") and recorder._device is not None and recorder._device.type == 'multimeter':
#                    self._device = recorder._device
#                    break
#            if self._device is None:
#                self._device = RecordingDevice("multimeter", to_memory)
#            self._device.add_variables(*VARIABLE_MAP.get(self.variable, [self.variable]))

    def _record(self, variable, new_ids):
        """
        Add the cells in `new_ids` to the set of recorded cells for the given
        variable. Since a given node can only be recorded from by one multimeter
        (http://www.nest-initiative.org/index.php/Analog_recording_with_multimeter, 14/11/11)
        we record all analog variables for all requested cells. 
        """
        if variable == 'spikes':
            self._spike_detector.add_ids(new_ids)
        else:
            self._multimeter.add_variable(variable)
            self._multimeter.add_ids(new_ids)

    #def _reset(self):
    #    """ """
    #    try:
    #        simulator.recording_devices.remove(self._device)
    #    except ValueError:
    #        pass
    #    
    #    if self._device != None:
    #          recorders_to_reset=[]
    #          for recorder in self.population.recorders.values():
    #              if hasattr(recorder, "_device") and recorder._device == self._device:
    #                 recorders_to_reset.append(recorder)
    #          for recorder in recorders_to_reset:
    #              recorder._device = None 
    #    self._create_device()

    def _get(self, variables, gather=False, filter_ids=None):
        """Return the recorded data as a Neo `Block`."""
        always_local = (hasattr(self.population.celltype, 'always_local') and self.population.celltype.always_local)
        data = neo.Block()
        # TODO: add cached data from previous runs
        data.segments.append(self._get_current_segment(filter_ids=filter_ids, variables=variables))
        return data
        
    def _get_current_segment(self, filter_ids=None, variables='all'):
        segment = neo.Segment(name=self.population.label,
                              description=self.population.describe(),
                              rec_datetime=datetime.now()) # would be nice to get the time at the start of the recording, not the end
        variables_to_include = set(self.recorded.keys())
        if variables is not 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        for variable in variables_to_include:
            if variable == 'spikes':
                spike_times = self._spike_detector.get_spiketimes(self.filter_recorded('spikes', filter_ids))
                segment.spiketrains = [
                    neo.SpikeTrain(spike_times[id],
                                   t_stop=simulator.state.t*pq.ms,
                                   units='ms',
                                   source_population=self.population.label,
                                   source_id=int(id)) # index?
                    for id in self.filter_recorded('spikes', filter_ids)]
            else:
                data = self._multimeter.get_data(VARIABLE_MAP.get(variable, variable),
                                                 self.filter_recorded(variable, filter_ids))
                segment.analogsignals = [
                    neo.AnalogSignal(data[id],
                                     units=recording.UNITS_MAP.get(variable, 'dimensionless'),
                                     t_start=simulator.state.t_start*pq.ms,
                                     sampling_period=simulator.state.dt*pq.ms,
                                     name=variable,
                                     source_population=self.population.label,
                                     source_id=int(id))
                    for id in self.filter_recorded(variable, filter_ids)]
        return segment

    def _local_count(self, filter):
        N = {}
        if self._device.in_memory():
            events = nest.GetStatus(self._device.device, 'events')[0]
            for id in self.filter_recorded(filter):
                mask = events['senders'] == int(id)
                N[int(id)] = len(events['times'][mask])
        else:
            spikes = self._get(gather=False, compatible_output=False,
                               filter=filter)
            for id in self.filter_recorded(filter):
                N[int(id)] = 0
            ids   = numpy.sort(spikes[:,0].astype(int))
            idx   = numpy.unique(ids)
            left  = numpy.searchsorted(ids, idx, 'left')
            right = numpy.searchsorted(ids, idx, 'right')
            for id, l, r in zip(idx, left, right):
                N[id] = r-l
        return N
