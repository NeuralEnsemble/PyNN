"""
Current source classes for the Arbor module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.
    ACSource           -- a sine modulated current.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import arbor
import numpy as np
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.morphology import MorphologyFilter
from pyNN.arborproto import simulator


class ArborCurrentSource(StandardCurrentSource):

    def __init__(self, **parameters):
        self._devices = []
        self.cell_list = []
        self._amplitudes = None
        self._times = None
        self._h_iclamps = {}
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)
        #  simulator.state.current_sources.append(self)

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for name, value in parameters.items():
            if name == "amplitudes":  # key used only by StepCurrentSource
                step_times = parameters["times"].value
                step_amplitudes = parameters["amplitudes"].value
                step_times, step_amplitudes = self._check_step_times(
                    step_times, step_amplitudes, simulator.state.dt)
                parameters["times"].value = step_times
                parameters["amplitudes"].value = step_amplitudes
            if isinstance(value, Sequence):  # this shouldn't be necessary, but seems to prevent a segfault
                value = value.value
            object.__setattr__(self, name, value)
        # self._reset()

    # def _reset(self):
    #     if self._is_computed:
    #         self._amplitudes = None
    #         self._times = None
    #         self._generate()
    #     for iclamp in self._h_iclamps.values():
    #         self._update_iclamp(iclamp, 0.0)    # send tstop = 0.0 on _reset()

    def inject_into(self, cells, location=None):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        # print(cells)
        for i in range(cells.all_cells.size):
            cells.all_cells[i]._cell._decor.place('"soma_midpoint"',  # '"{}"'.format(location),
                                                  arbor.iclamp(self.start, self.start+self.stop,
                                                               current=self.amplitude),
                                                  "iclamp"+str(i))


class DCSource(ArborCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude'),
        ('start', 'start'),
        ('stop', 'stop')
    )

    def inject_into(self, cells, location=None):
        # __doc__ = StandardCurrentSource.inject_into.__doc__
        # for i in range(cells.all_cells.size):
        #     cells.all_cells[i]._cell._decor.place('"soma_midpoint"',  # '"{}"'.format(location),
        #                                           arbor.iclamp(self.start, self.start+self.stop,
        #                                                        current=self.amplitude),
        #                                           "iclamp"+str(i))
        # step_current.inject_into(cells[0:1], location="soma")
        # step_current.inject_into(cells[1:2], location=random_section(apical_dendrites()))
        #
        for i in range(cells.all_cells.size):
            cells.all_cells[i]._cell._decor.place('"root"', arbor.iclamp(10, 1, current=2), "iclamp0")
            cells.all_cells[i]._cell._decor.place('"root"', arbor.iclamp(30, 1, current=2), "iclamp1")
            cells.all_cells[i]._cell._decor.place('"root"', arbor.iclamp(50, 1, current=2), "iclamp2")
            cells.all_cells[i]._cell._decor.place('"axon_terminal"', arbor.spike_detector(-10), "inj_detector")
        return cells



class StepCurrentSource(ArborCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes', 'amplitudes'),
        ('times', 'times')
    )


class ACSource(ArborCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude'),
        ('start', 'start'),
        ('stop', 'stop'),
        ('frequency', 'frequency'),
        ('offset', 'offset'),
        ('phase', 'phase')
    )


class NoisyCurrentSource(ArborCurrentSource, electrodes.NoisyCurrentSource):
    translations = build_translations(
        ('mean', 'mean'),
        ('start', 'start'),
        ('stop', 'stop'),
        ('stdev', 'stdev'),
        ('dt', 'dt')
    )
