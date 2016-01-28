# encoding: utf-8
"""
Standard cells for the NeuroML module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import cells, synapses, electrodes, build_translations, StandardCurrentSource
from .simulator import state
import logging

import neuroml

logger = logging.getLogger("PyNN")

def add_params(pynn_cell, nml_cell):
    for param in pynn_cell.simple_parameters():
        value = float(pynn_cell.parameter_space[param].base_value)
        nml_param = param  # .lower() if (not 'tau_syn' in param and not 'e_rev' in param) else param
        print("Adding param: %s = %s as %s for cell %s"%(param, value, nml_param, nml_cell.id))
        nml_cell.__setattr__(nml_param, value)
        
        nml_cell.__setattr__('v_init', pynn_cell.default_initial_values['v'])
    

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
        cell = neuroml.IF_curr_alpha(id="%s_%s"%(self.__class__.__name__, population.label))
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
        cell = neuroml.IF_curr_exp(id="%s_%s"%(self.__class__.__name__, population.label))
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
        cell = neuroml.IF_cond_alpha(id="%s_%s"%(self.__class__.__name__, population.label))
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
        cell = neuroml.IF_cond_exp(id="%s_%s"%(self.__class__.__name__, population.label))
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
        cell = neuroml.HH_cond_exp(id="%s_%s"%(self.__class__.__name__, population.label))
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
        cell = neuroml.SpikeSourcePoisson(id="%s_%s"%(self.__class__.__name__, population.label))
        nml_doc.SpikeSourcePoisson.append(cell)
        add_params(self, cell)
        return cell.id

class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'SPIKE_TIMES'),
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        pass
        

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
        cell = neuroml.IF_curr_exp(id="%s_%s"%(self.__class__.__name__, population.label))
        nml_doc.IF_curr_exp.append(cell)
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
        cell = neuroml.EIF_cond_exp_isfa_ista(id="%s_%s"%(self.__class__.__name__, population.label))
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
        cell = neuroml.Izhikevich(id="%s_%s"%(self.__class__.__name__, population.label))
        nml_doc.Izhikevich.append(cell)
        add_params(self, cell)
        return cell.id
        
    
class NeuroMLCurrentSource(object):
    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        pass


class DCSource(NeuroMLCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop')
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class StepCurrentSource(NeuroMLCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


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
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


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


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
        ('U', 'UU'),
        ('tau_rec', 'TAU_REC'),
        ('tau_facil', 'TAU_FACIL'),
        ('u0', 'U0'),
        ('x0', 'X' ),
        ('y0', 'Y')
    )
    
    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY')
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()
    

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
        ('mu_plus',   'muLTP'),
        ('mu_minus',  'muLTD'),
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tauLTP'),
        ('tau_minus', 'tauLTD'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    
    def add_to_nml_doc(self, nml_doc, population):
        raise NotImplementedError()
        
