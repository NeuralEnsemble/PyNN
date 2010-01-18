import tempfile
import os
import numpy
import logging
import nest
from pyNN import recording, common
from pyNN.nest import simulator

RECORDING_DEVICE_NAMES = {'spikes': 'spike_detector',
                          'v':      'voltmeter',
                          'gsyn':   'conductancemeter'}

logger = logging.getLogger("PyNN")

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    
    scale_factors = {'spikes': 1,
                     'v': 1,
                     'gsyn': 0.001} # units conversion
    
    def __init__(self, variable, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        assert variable in RECORDING_DEVICE_NAMES
        recording.Recorder.__init__(self, variable, population, file)       
        # we defer creating the actual device until it is needed.
        self._device = None
        self._local_files_merged = False
        self._gathered = False
        
    def in_memory(self):
        """Determine whether data is being recorded to memory."""
        return nest.GetStatus(self._device, 'to_memory')[0]

    def _create_device(self):
        """Create a NEST recording device."""
        device_name = RECORDING_DEVICE_NAMES[self.variable]
        self._device = nest.Create(device_name)
        device_parameters = {"withgid": True, "withtime": True}
        if self.variable != 'spikes':
            device_parameters["interval"] = common.get_time_step()
        if self.file is False:
            device_parameters.update(to_file=False, to_memory=True)
        else: # (includes self.file is None)
            device_parameters.update(to_file=True, to_memory=False)
        try:
            nest.SetStatus(self._device, device_parameters)
        except nest.hl_api.NESTError, e:
            raise nest.hl_api.NESTError("%s. Parameter dictionary was: %s" % (e, device_parameters))

    def _record(self, new_ids):
        """Called by record()."""            
        if self._device is None:
            self._create_device()       
        device_name = nest.GetStatus(self._device, "model")[0]
        if device_name == "spike_detector":
            nest.ConvergentConnect(new_ids, self._device, model='static_synapse')
        elif device_name in ('voltmeter', 'conductancemeter'):
            nest.DivergentConnect(self._device, new_ids, model='static_synapse')
        else:
            raise Exception("%s is not a valid recording device" % device_name)
    
    def _add_initial_and_scale(self, data):
        """
        Add initial values (NEST does not record the value at t=0), and scale the
        data to give appropriate units.
        """
        if self.variable == 'v':
            try:
                initial = [[id, 0.0, id.v_init] for id in self.recorded]
            except common.NonExistentParameterError:
                initial = [[id, 0.0, id.v_rest] for id in self.recorded]
        elif self.variable == 'gsyn':
            initial = [[id, 0.0, 0.0, 0.0] for id in self.recorded]
        else:
            initial = None
        if initial and self.recorded:
            data = numpy.concatenate((initial, data))
        # scale data
        scale_factor = Recorder.scale_factors[self.variable]
        if scale_factor != 1:
            data *= scale_factor
        return data
    
    def _events_to_array(self, events):
        """
        Transform the NEST events dictionary (when recording to memory) to a
        Numpy array.
        """
        ids = events['senders']
        times = events['times']
        if self.variable == 'spikes':
            data = numpy.array((ids, times)).T
        elif self.variable == 'v':
            data = numpy.array((ids, times, events['potentials'])).T
        elif self.variable == 'gsyn':
            data = numpy.array((ids, times, events['exc_conductance'], events['inh_conductance'])).T
        return data
                   
    def _read_data_from_memory(self, gather, compatible_output):
        """
        Return memory-recorded data.
        
        `gather` -- if True, gather data from all MPI nodes.
        `compatible_output` -- if True, transform the data into the PyNN
                               standard format.
        """
        data = nest.GetStatus(self._device,'events')[0] # only for spikes?
        if compatible_output:
            data = self._events_to_array(data)
            data = self._add_initial_and_scale(data)
        if gather and simulator.state.num_processes > 1:
            data = recording.gather(data)
        return data
                     
    def _read_data(self, gather, compatible_output):
        """
        Return file-recorded data.
        
        `gather` -- if True, gather data from all MPI nodes.
        `compatible_output` -- if True, transform the data into the PyNN
                               standard format.
                               
        Gathered data is cached, so the MPI communication need only be done
        once, even if the method is called multiple times.
        """
        # what if the method is called with different values of
        # `compatible_output`? Need to cache these separately.
        if gather and simulator.state.num_processes > 1:
            if self._gathered:
                logger.debug("Loading previously gathered data from cache")
                self._gathered_file.seek(0)
                data = numpy.load(self._gathered_file)
            else:
                local_data = self._read_local_data(compatible_output)
                if self.population and hasattr(self.population.celltype, 'always_local') and self.population.celltype.always_local:
                    data = local_data # for always_local cells, no need to gather
                else:
                    data = recording.gather(local_data)
                logger.debug("Caching gathered data")
                self._gathered_file = tempfile.TemporaryFile()
                numpy.save(self._gathered_file, data)
                self._gathered = True
            return data
        else:
            return self._read_local_data(compatible_output)
                        
    def _read_local_data(self, compatible_output):
        """
        Return file-recorded data from the local MPI node, merging data from
        different threads if applicable.
        
        The merged data is cached, to avoid the overhead of re-merging if the
        method is called again.
        """
        # what if the method is called with different values of
        # `compatible_output`? Need to cache these separately.
        if self._local_files_merged:
            self._merged_file.seek(0)
            data = numpy.load(self._merged_file)
        else:
            if "filename" not in nest.GetStatus(self._device)[0]: # indicates that run() has not been called.
                raise common.NothingToWriteError("No data. Have you called run()?")
            if simulator.state.num_threads > 1:
                nest_files = []
                for nest_thread in range(1, simulator.state.num_threads):
                    addr = nest.GetStatus(self._device, "address")[0]
                    addr.append(nest_thread)
                    nest_files.append(nest.GetStatus([addr], "filename")[0])
            else:
                nest_files = [nest.GetStatus(self._device, "filename")[0]]
            # possibly we can just keep on saving to the end of self._merged_file, instead of concatenating everything in memory
            logger.debug("Concatenating data from the following files: %s" % ", ".join(nest_files))
            non_empty_nest_files = [filename for filename in nest_files if os.stat(filename).st_size > 0]
            if len(non_empty_nest_files) > 0:
                data_list = [numpy.loadtxt(nest_file) for nest_file in non_empty_nest_files]
                data = numpy.concatenate(data_list)
            if len(non_empty_nest_files) == 0 or data.size == 0:
                ncol = len(Recorder.formats[self.variable].split())
                data = numpy.empty([0, ncol])
            if compatible_output:
                data = self._add_initial_and_scale(data)
            self._merged_file = tempfile.TemporaryFile()
            numpy.save(self._merged_file, data)
            self._local_files_merged = True
        return data
    
    def _get(self, gather=False, compatible_output=True):
        """Return the recorded data as a Numpy array."""
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to return")
        
        if self.in_memory():
            data = self._read_data_from_memory(gather, compatible_output)
        else: # in file
            data = self._read_data(gather, compatible_output)
        return data
    
    def write(self, file=None, gather=False, compatible_output=True):
        """Write recorded data to file."""
        if self._device is None:
            raise common.NothingToWriteError("%s not recorded from any cells, so no data to write to file." % self.variable)
        recording.Recorder.write(self, file, gather, compatible_output)

    def _local_count(self):
        N = {}
        if self.in_memory():
            events = nest.GetStatus(self._device, 'events')
            for id in self.recorded:
                mask = events['senders'] == int(id)
                N[id] = events['times'][mask].count()
        else:
            spikes = self._get(gather=False, compatible_output=False)
            for id in spikes[:,0].astype(int):
                assert id in self.recorded
                if id in N:
                    N[id] += 1
                else:
                    N[id] = 1
            for id in self.recorded:
                if id not in N:
                    N[id] = 0
        return N
    
simulator.Recorder = Recorder # very inelegant. Need to rethink the module structure