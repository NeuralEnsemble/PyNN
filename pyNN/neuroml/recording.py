"""

Export of PyNN models to NeuroML 2

Contact Padraig Gleeson for more details

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

# flake8: noqa

import numpy as np
from pyNN import recording
from . import simulator

import logging
logger = logging.getLogger("PyNN_NeuroML")

class Recorder(recording.Recorder):
    _simulator = simulator

    def __init__(self, population, file=None):
        super(Recorder, self).__init__(population, file=file)
        self.event_output_files = []
        self.displays = []
        self.output_files = []

    def _record(self, variable, new_ids, sampling_interval=None):

        lems_sim = simulator._get_lems_sim()

        for id in new_ids:
            if variable == 'v':
                logger.debug("Recording var: %s; %s; %s"%(variable, id, id.parent))
                pop_id = id.parent.label
                celltype = id.parent.celltype.__class__.__name__
                disp_id = '%s_%s'%(pop_id,variable)
                of_id = 'OF_%s'%disp_id
                index = id.parent.id_to_index(id)

                if not disp_id in self.displays:
                    lems_sim.create_display(disp_id, '%s %s'%(pop_id,variable), "-70", "10")
                    self.displays.append(disp_id)

                if not of_id in self.output_files:
                    lems_sim.create_output_file(of_id, "%s.dat"%disp_id)
                    self.output_files.append(of_id)

                #quantity = "%s/%i/%s/%s"%(pop_id,index,id.celltype.__class__.__name__,variable)
                quantity = "%s/%i/%s_%s/%s"%(pop_id,index,celltype,pop_id,variable)
                lems_sim.add_line_to_display(disp_id, '%s %s: cell %s'%(pop_id,variable,id), quantity, "1mV")
                lems_sim.add_column_to_output_file(of_id, quantity.replace('/','_'), quantity)

            elif variable == 'spikes':
                logger.debug("Recording spike: %s; %s; %s"%(variable, id, id.parent))
                pop_id = id.parent.label
                celltype = id.parent.celltype.__class__.__name__
                index = id.parent.id_to_index(id)

                eof0 = 'Spikes_file_%s'%pop_id

                if not eof0 in self.event_output_files:
                    lems_sim.create_event_output_file(eof0, "%s.spikes"%pop_id, format='TIME_ID')
                    self.event_output_files.append(eof0)

                lems_sim.add_selection_to_event_output_file(eof0, index, "%s/%i/%s_%s"%(pop_id,index,celltype,pop_id), 'spike')



    def _get_spiketimes(self, id, clear=False):

        if hasattr(id, "__len__"):
            spks = {}
            for i in id:
                spks[i] = np.array([i, i + 5], dtype=float) % self._simulator.state.t
            return spks
        else:
            return np.array([id, id + 5], dtype=float) % self._simulator.state.t

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        n_samples = int(round(self._simulator.state.t/self._simulator.state.dt)) + 1
        times = None
        return np.vstack([np.random.uniform(size=n_samples) for id in ids]).T, times

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
        self.displays = []
        self.output_files = []
        self.event_output_files = []
