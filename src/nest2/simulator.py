import nest
from pyNN import common, recording
import logging
import numpy
import os

RECORDING_DEVICE_NAMES = {'spikes': 'spike_detector',
                          'v': 'voltmeter',
                          'conductance': 'conductancemeter'}
recorder_list = []

def _merge_files(recorder, gather):
    """
    Combine data from multiple files (one per thread and per process) into a single file.
    Returns the filename of the merged file.
    """
    status = nest.GetStatus([0])[0]
    local_num_threads = status['local_num_threads']
    node_list = range(nest.GetStatus([0], "num_processes")[0])

    # Combine data from different threads to the zeroeth thread
    merged_filename = nest.GetStatus(recorder, "filename")[0]

    if local_num_threads > 1:
        for nest_thread in range(1, local_num_threads):
            addr = nest.GetStatus(recorder, "address")[0]
            addr.append(nest_thread)
            nest_filename = nest.GetStatus([addr], "filename")[0]
            system_line = 'cat %s >> %s' % (nest_filename, merged_filename)
            os.system(system_line)
            os.remove(nest_filename)
    if gather and len(node_list) > 1:
        if rank() == 0:
            raise Exception("gather not yet implemented")
    return merged_filename


class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    formats = {'spikes': 'id t',
               'v': 'id t v',
               'conductance':'id t ge gi'}
    
    def __init__(self, variable, population=None, file=None):
        """
        `file` should be one of:
            a file-name,
            `None` (write to a temporary file)
            `False` (write to memory).
        """
        assert variable in RECORDING_DEVICE_NAMES
        self.variable = variable
        self.file = file
        self.population = population # needed for writing header information
        self.recorded = set([])        
        # we defer creating the actual device until it is needed.
        self._device = None

    def _create_device(self):
        device_name = RECORDING_DEVICE_NAMES[self.variable]
        self._device = nest.Create(device_name)
        device_parameters = {"withgid": True, "withtime": True}
        if self.variable != 'spikes':
            device_parameters["interval"] = common.get_time_step()
        if self.file is False:
            device_parameters.update(to_file=False, to_memory=True)
        else: # (includes self.file is None)
            device_parameters.update(to_file=True, to_memory=False)
        # line needed for old version of nest 2.0
        #device_parameters.pop('to_memory')
        nest.SetStatus(self._device, device_parameters)

    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        if self._device is None:
            self._create_device()
        ids = set(ids)
        new_ids = list( ids.difference(self.recorded) )
        self.recorded = self.recorded.union(ids)
        
        device_name = nest.GetStatus(self._device, "model")[0]
        if device_name == "spike_detector":
            nest.ConvergentConnect(new_ids, self._device, model='static_synapse')
        elif device_name in ('voltmeter', 'conductancemeter'):
            nest.DivergentConnect(self._device, new_ids, model='static_synapse')
        else:
            raise Exception("%s is not a valid recording device" % device_name)
    
    def get(self, gather=False, compatible_output=True):
        """Returns the recorded data."""
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to return")
        if nest.GetStatus(self._device, 'to_file')[0]:
            if 'filename' in nest.GetStatus(self._device)[0]:
                nest_filename = _merge_files(self._device, gather)
                data = recording.readArray(nest_filename, sepchar=None)
            else:
                data = numpy.array([])
            #os.remove(nest_filename)
            if data.size > 0:
                # the following returns indices, not IDs. I'm not sure this is what we want.
                if self.population is not None:
                    padding = self.population.cell.flatten()[0]
                else:
                    padding = 0
                data[:,0] = data[:,0] - padding
            else:
                ncol = len(Recorder.formats[self.variable].split())
                data = numpy.empty([0, ncol])
        elif nest.GetStatus(self._device,'to_memory')[0]:
            data = nest.GetStatus(self._device,'events')[0]
            data = recording.convert_compatible_output(data, self.population, self.variable,compatible_output)
        return data
    
    def _get_header(self, file_name):
        header = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for line in f:
                if line[0] == '#':
                    key, value = line[1:].split("=")
                    header[key.strip()] = value.strip()
        else:
            logging.warning("File %s does not exist, so could not get header." % file_name)
        return header
    
    def _strip_header(self, input_file_name, output_file_name):
        if os.path.exists(input_file_name):
            fin = open(input_file_name, 'r')
            fout = open(output_file_name, 'a')
            for line in fin:
                if line[0] != '#':
                    fout.write(line)
            fin.close()
            fout.close()
    
    def write(self, file=None, gather=False, compatible_output=True):
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to write to file.")
        user_file = file or self.file
        if isinstance(user_file, basestring):
            if common.num_processes() > 1:
                user_file += '.%d' % rank()
            recording.rename_existing(user_file)
        logging.debug("Recorder is writing '%s' to file '%s' with gather=%s and compatible_output=%s" % (self.variable,
                                                                                                         user_file,
                                                                                                         gather,
                                                                                                         compatible_output))
        # what if the data was only saved to memory?
        if self.file is not False:
            nest_filename = _merge_files(self._device, gather)
            if compatible_output:
                # We should do the post processing (i.e the compatible output) in a distributed
                # manner to speed up the thing. The only problem that will arise is the headers, 
                # that should be taken into account to be really clean. Otherwise, if the # symbol
                # is escaped while reading the file, there is no problem
               recording.write_compatible_output(nest_filename, user_file,
                                                 self.variable,
                                                 Recorder.formats[self.variable],
                                                 self.population, common.get_time_step())
            else:
                if isinstance(user_file, basestring):
                    os.system('cat %s > %s' % (nest_filename, user_file))
                elif hasattr(user_file, 'write'):
                    nest_file = open(nest_filename)
                    user_file.write(nest_file.read())
                    nest_file.close()
                else:
                    raise Exception('Must provide a filename or an open file')
            np = common.num_processes()
            logging.debug("num_processes() = %s" % np)
            if gather == True and common.rank() == 0 and np > 1:
                root_file = file or self.filename
                logging.debug("Gathering files generated by different nodes into %s" % root_file)
                n_cells = 0
                recording.rename_existing(root_file)
                for node in xrange(np):
                    node_file = root_file + '.%d' % node
                    logging.debug("node_file = %s" % node_file)
                    if os.path.exists(node_file):
                        # merge headers
                        header = self._get_header(node_file)
                        logging.debug(str(header))
                        if header.has_key('n'):
                            n_cells += int(header['n'])
                        #os.system('cat %s >> %s' % (node_file, root_file))
                        self._strip_header(node_file, root_file)
                        os.system('rm %s' % node_file)
                # write header for gathered file
                f = open("tmp_header", 'w')
                header['n'] = n_cells
                for k,v in header.items():
                    f.write("# %s = %s\n" % (k,v))
                f.close()
                os.system('cat %s >> tmp_header' % root_file)
                os.system('mv tmp_header %s' % root_file)
            # don't want to remove nest_filename at this point in case the user wants to access the data
            # a second time (e.g. with both getSpikes() and printSpikes()), but we should
            # maintain a list of temporary files to be deleted at the end of the simulation
        else:
            raise Exception("Writing to file not yet supported for data recorded only to memory.")
