"""
Standard electrodes for the NeuroML module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

# flake8: noqa

from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.neuroml.simulator import _get_nml_doc, _get_main_network
import logging
from pyNN.parameters import ParameterSpace, Sequence

import neuroml

logger = logging.getLogger("PyNN_NeuroML")

current_sources = []


class NeuroMLCurrentSource(StandardCurrentSource):


    def __init__(self, **parameters):
        super(StandardCurrentSource, self).__init__(**parameters)
        global current_sources
        self.cell_list = []
        self.indices   = []
        self.ind = len(current_sources) # Todo use self.indices instead...
        current_sources.append(self)
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)



    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for name, value in parameters.items():
            if isinstance(value, Sequence):
                value = value.value
            object.__setattr__(self, name, value)

    def _get_input_list(self, stim_id, pop):
        id="Input_%s"%(stim_id)

        self.network = _get_main_network()
        for il in self.network.input_lists:
            if il.id ==id:
                return il
        input_list = neuroml.InputList(id=id,
                             component=stim_id,
                             populations=pop)

        self.network.input_lists.append(input_list)

        return input_list


    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__

        logger.debug("%s injecting into: %s"%(self.__class__.__name__, cells))

        self.nml_doc = _get_nml_doc()
        id = self.add_to_nml_doc(self.nml_doc, cells)



        for cell in cells:
            pop_id = cell.parent.label
            index = cell.parent.id_to_index(cell)
            celltype = cell.parent.celltype.__class__.__name__
            logger.debug("Injecting: %s to %s (%s[%s])"%(id, cell, pop_id, index))

            input_list = self._get_input_list(id, pop_id)

            input = neuroml.Input(id=len(input_list.input),
                              target="../%s/%i/%s_%s"%(pop_id, index, celltype, pop_id),
                              destination="synapses")
            input_list.input.append(input)

    def get_id_for_nml(self, cells):
        return "%s_%s_%s"%(self.__class__.__name__, cells.label if hasattr(cells, 'label') else self.__class__.__name__, self.ind)



class DCSource(NeuroMLCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop')
    )

    def add_to_nml_doc(self, nml_doc, cells):
        id=self.get_id_for_nml(cells)
        pg = neuroml.PulseGenerator(id=id,
                                      delay='%sms'%self.start,
                                      duration='%sms'%(self.stop-self.start),
                                      amplitude='%snA'%self.amplitude)

        found = False
        for pg in nml_doc.pulse_generators:
            if pg.id==id: found = True
        if not found:
            nml_doc.pulse_generators.append(pg)
        return pg.id


class StepCurrentSource(NeuroMLCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )

    def add_to_nml_doc(self, nml_doc, cells):
        ci = neuroml.CompoundInput(id=self.get_id_for_nml(cells))

        num_steps = len(self.amplitudes)
        for i in range(num_steps):
            next_time = 1e9 if i==num_steps-1 else self.times[i+1]
            ci.pulse_generators.append(neuroml.PulseGenerator(id='step_%s'%i,delay='%sms'%self.times[i],duration='%sms'%(next_time-self.times[i]),amplitude='%snA'%self.amplitudes[i]))

        self.nml_doc = _get_nml_doc()
        self.nml_doc.compound_inputs.append(ci)
        return ci.id


class ACSource(NeuroMLCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop'),
        ('frequency',  'frequency'),
        ('offset',     'offset'),
        ('phase',      'phase')
    )

    def add_to_nml_doc(self, nml_doc, cells):

        ci = neuroml.CompoundInput(id=self.get_id_for_nml(cells))

        sg = neuroml.SineGenerator(id='SG_'+self.get_id_for_nml(cells),
                             delay='%sms'%self.start,
                             duration='%sms'%(self.stop-self.start),
                             amplitude='%snA'%self.amplitude,
                             period='%s s'%(1/float(self.frequency)),
                             phase=(3.14159265 * self.phase/180))

        pg = neuroml.PulseGenerator(id='PG_'+self.get_id_for_nml(cells),
                                      delay='%sms'%self.start,
                                      duration='%sms'%(self.stop-self.start),
                                      amplitude='%snA'%self.offset)

        ci.sine_generators.append(sg)
        ci.pulse_generators.append(pg)

        self.nml_doc = _get_nml_doc()
        self.nml_doc.compound_inputs.append(ci)
        return ci.id


class NoisyCurrentSource(NeuroMLCurrentSource, electrodes.NoisyCurrentSource):

    translations = build_translations(
        ('mean',  'mean'),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'stdev'),
        ('dt',    'dt')
    )

    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()
