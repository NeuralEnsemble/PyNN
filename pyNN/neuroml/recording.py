import numpy
from pyNN import recording
from . import simulator

import logging
logger = logging.getLogger("PyNN_NeuroML")


class Recorder(recording.Recorder):
    _simulator = simulator
    
    displays = []
    output_files = []

    def _record(self, variable, new_ids, sampling_interval=None):
        
        lems_sim = simulator.get_lems_sim()
        
        for id in new_ids:
            if variable == 'v':
                logger.debug("Recording: %s; %s; %s"%(variable, id, id.parent))
                pop_id = id.parent.label
                disp_id = '%s_%s'%(pop_id,variable)
                of_id = 'output_%s'%disp_id
                index = id.parent.id_to_index(id)

                if not disp_id in self.displays:
                    lems_sim.create_display(disp_id, '%s %s'%(pop_id,variable), "-70", "10")
                    self.displays.append(disp_id)

                if not of_id in self.output_files:
                    lems_sim.create_output_file(of_id, "%s.dat"%of_id)
                    self.output_files.append(of_id)

                #quantity = "%s/%i/%s/%s"%(pop_id,index,id.celltype.__class__.__name__,variable)
                quantity = "%s[%i]/%s"%(pop_id,index,variable)
                lems_sim.add_line_to_display(disp_id, '%s %s: cell %s'%(pop_id,variable,id), quantity, "1mV")
                lems_sim.add_column_to_output_file(of_id, quantity, quantity)
            

    def _get_spiketimes(self, id):
        return numpy.array([id, id+5], dtype=float) % self._simulator.state.t

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        n_samples = int(round(self._simulator.state.t/self._simulator.state.dt)) + 1
        return numpy.vstack((numpy.random.uniform(size=n_samples) for id in ids)).T

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = 2
        else:
            raise Exception("Only implemented for spikes")
        return N

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
