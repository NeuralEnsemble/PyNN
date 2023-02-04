"""
Standard cells for the NeuroML module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

# flake8: noqa

from pyNN.standardmodels import cells, build_translations
import logging
from pyNN.random import RandomDistribution

import neuroml

logger = logging.getLogger("PyNN_NeuroML")


def add_params(pynn_cell, nml_cell, set_v_init=True):
    """Copy the parameters set in PyNN to the NeuroML equivalent"""
    for param in pynn_cell.simple_parameters():
        value_generator = pynn_cell.parameter_space[param].base_value
        #print(value_generator)
        # TODO: handle this....
        if isinstance(value_generator, RandomDistribution):
            print('*'*200+'\n\nRandom element in population! Not supported!!\n\n'+'*'*200)
            value = value_generator.next()
        else:
            value = float(value_generator)
        nml_param = param  # .lower() if (not 'tau_syn' in param and not 'e_rev' in param) else param
        logger.debug("Adding param: %s = %s as %s for cell %s"%(param, value, nml_param, nml_cell.id))
        nml_cell.__setattr__(nml_param, value)
    if set_v_init:
        nml_cell.__setattr__('v_init', pynn_cell.default_initial_values['v'])
        logger.debug("Adding param: %s = %s as %s for cell %s"%('v_init', pynn_cell.default_initial_values['v'], 'v_init', nml_cell.id))


class IF_curr_alpha(cells.IF_curr_alpha):
    __doc__ = cells.IF_curr_alpha.__doc__

    translations = build_translations(  # should add some computed/scaled parameters
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.IF_curr_alpha(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.IF_curr_alpha.append(cell)
        add_params(self, cell)
        return cell.id


class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__

    translations = build_translations(  # should add some computed/scaled parameters
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'T_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.IF_curr_exp(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.IF_curr_exp.append(cell)
        add_params(self, cell)
        return cell.id

class IF_cond_alpha(cells.IF_cond_alpha):
    __doc__ = cells.IF_cond_alpha.__doc__

    translations = build_translations(
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I')
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.IF_cond_alpha(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.IF_cond_alpha.append(cell)
        add_params(self, cell)
        return cell.id


class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I')
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.IF_cond_exp(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.IF_cond_exp.append(cell)
        add_params(self, cell)
        return cell.id

class IF_facets_hardware1(cells.IF_facets_hardware1):
    __doc__ = cells.IF_facets_hardware1.__doc__


class HH_cond_exp(cells.HH_cond_exp):
    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'GBAR_NA'),
        ('gbar_K',     'GBAR_K'),
        ('g_leak',     'G_LEAK'),
        ('cm',         'CM'),
        ('v_offset',   'V_OFFSET'),
        ('e_rev_Na',   'E_REV_NA'),
        ('e_rev_K',    'E_REV_K'),
        ('e_rev_leak', 'E_REV_LEAK'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('i_offset',   'I_OFFSET'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.HH_cond_exp(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.HH_cond_exp.append(cell)
        add_params(self, cell)
        return cell.id

class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('start',    'START'),
        ('rate',     'INTERVAL',  "1000.0/rate",  "1000.0/INTERVAL"),
        ('duration', 'DURATION'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.SpikeSourcePoisson(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.SpikeSourcePoisson.append(cell)
        cell.start = '%sms'%self.parameter_space['start'].base_value
        cell.duration = '%sms'%self.parameter_space['duration'].base_value
        cell.rate = '%sHz'%self.parameter_space['rate'].base_value
        return cell.id

class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'SPIKE_TIMES'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.SpikeArray(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        index=0
        spikes = self.parameter_space['spike_times']

        for spike_time in spikes.base_value.value:
            cell.spikes.append(neuroml.Spike(id=index, time='%sms'%spike_time))
            index+=1
        nml_doc.spike_arrays.append(cell)
        return cell.id

class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'CM'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('v_spike',    'V_SPIKE'),
        ('v_reset',    'V_RESET'),
        ('v_rest',     'V_REST'),
        ('tau_m',      'TAU_M'),
        ('i_offset',   'I_OFFSET'),
        ('a',          'A'),
        ('b',          'B'),
        ('delta_T',    'DELTA_T'),
        ('tau_w',      'TAU_W'),
        ('v_thresh',   'V_THRESH'),
        ('e_rev_E',    'E_REV_E'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.EIF_cond_alpha_isfa_ista(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.EIF_cond_alpha_isfa_ista.append(cell)
        add_params(self, cell)
        return cell.id

class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'CM'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('v_spike',    'V_SPIKE'),
        ('v_reset',    'V_RESET'),
        ('v_rest',     'V_REST'),
        ('tau_m',      'TAU_M'),
        ('i_offset',   'I_OFFSET'),
        ('a',          'A'),
        ('b',          'B'),
        ('delta_T',    'DELTA_T'),
        ('tau_w',      'TAU_W'),
        ('v_thresh',   'V_THRESH'),
        ('e_rev_E',    'E_REV_E'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.EIF_cond_exp_isfa_ista(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.EIF_cond_exp_isfa_ista.append(cell)
        add_params(self, cell)
        return cell.id

class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'I_e'),
    )
    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling

    def add_to_nml_doc(self, nml_doc, population):
        cell = neuroml.Izhikevich(id="%s_%s"%(self.__class__.__name__, population.label if population else '0'))
        nml_doc.Izhikevich.append(cell)
        add_params(self, cell)
        return cell.id
