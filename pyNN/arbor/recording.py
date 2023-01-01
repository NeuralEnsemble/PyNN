# encoding: utf-8
"""
Arbor implementation of the PyNN API.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy as np
from pyNN import recording
from pyNN.arbor import simulator
import arbor


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        # cells.record('spikes')
        # cells.record(['na.m', 'na.h', 'kdr.n'], locations={'soma': 'soma'})
        # cells.record('v', locations={'soma': 'soma', 'dendrite': 'dendrite'})
        if variable.name == 'spikes': # record from all cells
            # for id in new_ids:
            #     if id._cell.rec is not None:
            #         id._cell.rec.record(id._cell.spike_times)
            #     else:  # SpikeSourceArray
            #         id._cell.recording = True
            for id in new_ids:
                # id._cell.__arbor_labels.update({'soma_midpoint': '(location 0 0.5)'})
                id._cell.__decor.place('"soma_midpoint"', arbor.spike_detector(-10), "detector")
        else:
            self.sampling_interval = sampling_interval or self._simulator.state.dt
            for id in new_ids:
                self._record_state_variable(id._cell, variable)
        # # (2) Define the soma and its midpoint
        # labels = arbor.label_dict({'soma': '(tag 1)',
        #                            'midpoint': '(location 0 0.5)'})
        #
        # # (3) Create and set up a decor object
        # decor = arbor.decor()
        # decor.set_property(Vm=-40)
        # decor.paint('"soma"', arbor.density('hh'))
        # decor.place('"midpoint"', arbor.iclamp(10, 2, 0.8), "iclamp")
        # decor.place('"midpoint"', arbor.spike_detector(-10), "detector")

    def _record_state_variable(self, cell, variable):
        if variable.location is None:
            # if hasattr(cell, 'recordable') and variable in cell.recordable:
            #     hoc_var = cell.recordable[variable]
            if variable.name == 'v':
                #hoc_var = cell.source_section(0.5)._ref_v  # or use "seg.v"?
                if isinstance(cell, arbor._arbor.single_cell_model): # arbor._arbor.cable_cell
                    cell.probe('voltage', '"midpoint"', frequency=1/self.sampling_interval)
                else:
                    if "soma_midpoint" in cell.__arbor_labels:
                        cable_cell = arbor.cable_cell(cell.__arbor_tree, cell.__arbor_labels, cell.__decor)
                    else:
                        # cell.__arbor_labels.update({'soma_midpoint': '(location 0 0.5)'})
                        cable_cell = arbor.cable_cell(cell.__arbor_tree, cell.__arbor_labels, cell.__decor)
                    cell = arbor.single_cell_model(cable_cell) # change arbor.single_cell_model to custom recipe?
                    cell.probe('voltage', '"midpoint"', frequency=1/self.sampling_interval) # 1/self.sampling_interval
            # elif variable.name == 'gsyn_exc':
            #     hoc_var = cell.esyn._ref_g
            # elif variable.name == 'gsyn_inh':
            #     hoc_var = cell.isyn._ref_g
            # else:
            #     source, var_name = self._resolve_variable(cell, variable.name)
            #     hoc_var = getattr(source, "_ref_%s" % var_name)
            # hoc_vars = [hoc_var]
        else:
            if variable.name == "v":
                if isinstance(variable.location, str):
                    pass
                else:
                    for ky, value in variable.locations.items():
                        if value in cell.__arbor_labels.keys():
                            pass
                        else:
                            pass
            else:
                pass

    def _get_spiketimes(self, id):
        if hasattr(id, "__len__"):
            spks = {}
            for i in id:
                spks[i] = np.array([i, i + 5], dtype=float) % self._simulator.state.t
            return spks
        else:
            return np.array([id, id + 5], dtype=float) % self._simulator.state.t

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        n_samples = int(round(self._simulator.state.t / self._simulator.state.dt)) + 1
        return np.vstack((np.random.uniform(size=n_samples) for id in ids)).T

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable.name == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = 2
        else:
            raise Exception("Only implemented for spikes")
        return N

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
