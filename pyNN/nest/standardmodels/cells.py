"""
Standard cells for nest

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
from collections import defaultdict
import nest
from ... import errors
from ...parameters import ArrayParameter, LazyArray
from ...standardmodels import cells, build_translations
from .. import simulator

logger = logging.getLogger("PyNN")


class IF_curr_alpha(cells.IF_curr_alpha):

    __doc__ = cells.IF_curr_alpha.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0),  # I_e is in pA, i_offset in nA
    )
    variable_map = {'v': 'V_m', 'isyn_exc': 'I_syn_ex', 'isyn_inh': 'I_syn_in'}
    scale_factors = {'v': 1, 'isyn_exc': 0.001, 'isyn_inh': 0.001}
    nest_name = {"on_grid": "iaf_psc_alpha",
                 "off_grid": "iaf_psc_alpha"}
    standard_receptor_type = True


class IF_curr_exp(cells.IF_curr_exp):

    __doc__ = cells.IF_curr_exp.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0),  # I_e is in pA, i_offset in nA
    )
    variable_map = {'v': 'V_m', 'isyn_exc': 'I_syn_ex', 'isyn_inh': 'I_syn_in'}
    scale_factors = {'v': 1, 'isyn_exc': 0.001, 'isyn_inh': 0.001}
    nest_name = {"on_grid": 'iaf_psc_exp',
                 "off_grid": 'iaf_psc_exp_ps'}
    standard_receptor_type = True


class IF_curr_delta(cells.IF_curr_delta):

    __doc__ = cells.IF_curr_delta.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0),  # I_e is in pA, i_offset in nA
    )
    # extra parameters in the NEST model
    # V_min            mV      Absolute lower value for the membrane potenial
    # refractory_input boolean If true, do not discard input during
    #                      refractory period. Default: false
    nest_name = {"on_grid": 'iaf_psc_delta',
                 "off_grid": 'iaf_psc_delta_ps'}
    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling


class IF_cond_alpha(cells.IF_cond_alpha):

    __doc__ = cells.IF_cond_alpha.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0),  # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001}
    nest_name = {"on_grid": "iaf_cond_alpha",
                 "off_grid": "iaf_cond_alpha"}
    standard_receptor_type = True


class IF_cond_exp(cells.IF_cond_exp):

    __doc__ = cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0),  # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001}
    nest_name = {"on_grid": "iaf_cond_exp",
                 "off_grid": "iaf_cond_exp"}
    standard_receptor_type = True


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):

    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0),  # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_sfa',    'tau_sfa'),
        ('e_rev_sfa',  'E_sfa'),
        ('q_sfa',      'q_sfa'),
        ('tau_rr',     'tau_rr'),
        ('e_rev_rr',   'E_rr'),
        ('q_rr',       'q_rr')
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in',
                    'g_rr': 'g_rr', 'g_sfa': 'g_sfa'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001,
                     'g_rr': 0.001, 'g_sfa': 0.001}
    nest_name = {"on_grid": "iaf_cond_exp_sfa_rr",
                 "off_grid": "iaf_cond_exp_sfa_rr"}
    standard_receptor_type = True


class IF_facets_hardware1(cells.IF_facets_hardware1):

    __doc__ = cells.IF_facets_hardware1.__doc__

    # in 'iaf_cond_exp', the dimension of C_m is pF,
    # while in the pyNN context, cm is given in nF
    translations = build_translations(
        ('v_reset',    'V_reset'),
        ('v_rest',     'E_L'),
        ('v_thresh',   'V_th'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('g_leak',     'g_L')
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001}
    nest_name = {"on_grid": "iaf_cond_exp",
                 "off_grid": "iaf_cond_exp"}
    standard_receptor_type = True
    extra_parameters = {
        'C_m': 200.0,
        't_ref': 1.0,
        'E_ex': 0.0
    }


class HH_cond_exp(cells.HH_cond_exp):

    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'g_Na',  1000.0),  # uS --> nS
        ('gbar_K',     'g_K',   1000.0),
        ('g_leak',     'g_L',   1000.0),
        ('cm',         'C_m',   1000.0),  # nF --> pF
        ('v_offset',   'V_T'),
        ('e_rev_Na',   'E_Na'),
        ('e_rev_K',    'E_K'),
        ('e_rev_leak', 'E_L'),
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('i_offset',   'I_e',   1000.0),  # nA --> pA
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in',
                    'm': 'Act_m', 'n': 'Act_n', 'h': 'Inact_h'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001,
                     'm': 1, 'n': 1, 'h': 1}
    nest_name = {"on_grid": "hh_cond_exp_traub",
                 "off_grid": "hh_cond_exp_traub"}
    standard_receptor_type = True


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):

    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'C_m',       1000.0),  # nF -> pF
        ('tau_refrac', 't_ref'),
        ('v_spike',    'V_peak'),
        ('v_reset',    'V_reset'),
        ('v_rest',     'E_L'),
        ('tau_m',      'g_L',       "cm/tau_m*1000.0", "C_m/g_L"),
        ('i_offset',   'I_e',       1000.0),  # nA -> pA
        ('a',          'a'),
        ('b',          'b',         1000.0),  # nA -> pA.
        ('delta_T',    'Delta_T'),
        ('tau_w',      'tau_w'),
        ('v_thresh',   'V_th'),
        ('e_rev_E',    'E_ex'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_I',  'tau_syn_in'),
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in', 'w': 'w'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001, 'w': 0.001}
    nest_name = {"on_grid": "aeif_cond_alpha",
                 "off_grid": "aeif_cond_alpha"}
    standard_receptor_type = True


class SpikeSourcePoisson(cells.SpikeSourcePoisson):

    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('rate',     'rate'),
        ('start',    'start'),
        ('duration', 'stop',    "start+duration", "stop-start"),
    )
    nest_name = {"on_grid": 'poisson_generator',
                 "off_grid": 'poisson_generator_ps'}
    always_local = True
    uses_parrot = True
    extra_parameters = {
        'origin': 1.0
    }


def unsupported(parameter_name, valid_value):
    def error_if_invalid(**parameters):
        if parameters[parameter_name].base_value != valid_value:
            raise NotImplementedError(
                "The `{}` parameter is not supported in NEST".format(parameter_name))
        return valid_value
    return error_if_invalid


class SpikeSourcePoissonRefractory(cells.SpikeSourcePoissonRefractory):

    __doc__ = cells.SpikeSourcePoissonRefractory.__doc__

    translations = build_translations(
        ('rate',       'rate'),
        ('tau_refrac', 'dead_time'),
        ('start',    'UNSUPPORTED', unsupported('start', 0.0), None),
        ('duration', 'UNSUPPORTED', unsupported('duration', 1e10), None),
    )
    nest_name = {"on_grid": 'ppd_sup_generator',
                 "off_grid": 'ppd_sup_generator'}
    always_local = True
    uses_parrot = True
    extra_parameters = {
        'n_proc': 1,
        'frequency': 0.0,
    }


class SpikeSourceGamma(cells.SpikeSourceGamma):

    __doc__ = cells.SpikeSourceGamma.__doc__

    translations = build_translations(
        ('alpha',    'gamma_shape'),
        ('beta',     'rate',        'beta/alpha',   'gamma_shape * rate'),
        ('start',    'UNSUPPORTED', unsupported('start', 0.0), None),
        ('duration', 'UNSUPPORTED', unsupported('duration', 1e10), None),
    )
    nest_name = {"on_grid": 'gamma_sup_generator',
                 "off_grid": 'gamma_sup_generator'}
    always_local = True
    uses_parrot = True
    extra_parameters = {
        'n_proc': 1
    }


class SpikeSourceInhGamma(cells.SpikeSourceInhGamma):

    __doc__ = cells.SpikeSourceInhGamma.__doc__

    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('tbins',    'tbins'),
        ('start',    'start'),
        ('duration', 'stop',   "duration+start", "stop-start"),
    )
    nest_name = {"on_grid": 'inh_gamma_generator',
                 "off_grid":  'inh_gamma_generator'}
    always_local = True
    uses_parrot = True
    extra_parameters = {
        'origin': 1.0
    }


def adjust_spike_times_forward(spike_times):
    """
    Since this cell type requires parrot neurons, we have to adjust the
    spike times to account for the transmission delay from device to
    parrot neuron.
    """
    # todo: emit warning if any times become negative
    return spike_times - simulator.state.min_delay


def adjust_spike_times_backward(spike_times):
    """
    Since this cell type requires parrot neurons, we have to adjust the
    spike times to account for the transmission delay from device to
    parrot neuron.
    """
    return spike_times + simulator.state.min_delay


class SpikeSourceArray(cells.SpikeSourceArray):

    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'spike_times',
         adjust_spike_times_forward,
         adjust_spike_times_backward),
    )
    nest_name = {"on_grid": 'spike_generator',
                 "off_grid": 'spike_generator'}
    uses_parrot = True
    always_local = True


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):

    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'C_m',       1000.0),  # nF -> pF
        ('tau_refrac', 't_ref'),
        ('v_spike',    'V_peak'),
        ('v_reset',    'V_reset'),
        ('v_rest',     'E_L'),
        ('tau_m',      'g_L',       "cm/tau_m*1000.0", "C_m/g_L"),
        ('i_offset',   'I_e',       1000.0),  # nA -> pA
        ('a',          'a'),
        ('b',          'b',         1000.0),  # nA -> pA.
        ('delta_T',    'Delta_T'),
        ('tau_w',      'tau_w'),
        ('v_thresh',   'V_th'),
        ('e_rev_E',    'E_ex'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_I',  'tau_syn_in'),
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in', 'w': 'w'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001, 'w': 0.001}
    nest_name = {"on_grid": "aeif_cond_exp",
                 "off_grid": "aeif_cond_exp"}
    standard_receptor_type = True


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'I_e', 1000.0),
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in', 'u': 'U_m'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001, 'gsyn_inh': 0.001, 'u': 1}
    # todo: check 'u' scale factor
    nest_name = {"on_grid": "izhikevich",
                 "off_grid": "izhikevich"}
    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling


class GIF_cond_exp(cells.GIF_cond_exp):

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('cm',         'C_m',       1000.0),  # nF -> pF
        ('tau_m',      'g_L',       "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('v_reset',    'V_reset'),
        ('i_offset',   'I_e',       1000.0),  # nA -> pA
        ('delta_v',    'Delta_V'),
        ('v_t_star',   'V_T_star'),
        ('lambda0',    'lambda_0'),
        ('tau_eta',   'tau_stc'),
        ('tau_gamma', 'tau_sfa'),
        ('a_eta',     'q_stc',    1000.0),  # nA -> pA
        ('a_gamma',   'q_sfa'),
    )
    variable_map = {'v': 'V_m', 'gsyn_exc': 'g_ex', 'gsyn_inh': 'g_in',
                    'i_eta': 'I_stc', 'v_t': 'E_sfa'}
    scale_factors = {'v': 1, 'gsyn_exc': 0.001,
                     'gsyn_inh': 0.001, 'i_eta': 0.001, 'v_t': 1}
    nest_name = {"on_grid": "gif_cond_exp",
                 "off_grid": "gif_cond_exp"}
    standard_receptor_type = True


class LIF(cells.LIF):

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0),  # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0),  # I_e is in pA, i_offset in nA
    )
    variable_map = {'v': 'V_m'}
    scale_factors = {'v': 1}
    possible_models = set(["iaf_psc_alpha_multisynapse", "iaf_psc_exp_multisynapse"])
    standard_receptor_type = True


class AdExp(cells.AdExp):

    translations = build_translations(
        ('cm',         'C_m',       1000.0),  # nF -> pF
        ('tau_refrac', 't_ref'),
        ('v_spike',    'V_peak'),
        ('v_reset',    'V_reset'),
        ('v_rest',     'E_L'),
        ('tau_m',      'g_L',       "cm/tau_m*1000.0", "C_m/g_L"),
        ('i_offset',   'I_e',       1000.0),  # nA -> pA
        ('a',          'a'),
        ('b',          'b',         1000.0),  # nA -> pA.
        ('delta_T',    'Delta_T'),
        ('tau_w',      'tau_w'),
        ('v_thresh',   'V_th')
    )
    variable_map = {'v': 'V_m', 'w': 'w'}
    scale_factors = {'v': 1, 'w': 0.001}
    possible_models = set(["aeif_cond_alpha_multisynapse", "aeif_cond_beta_multisynapse"])
    standard_receptor_type = False


class PointNeuron(cells.PointNeuron):
    standard_receptor_type = False

    def get_receptor_type(self, name):
        return self.receptor_types.index(name) + 1  # port numbers start at 1

    @property
    def possible_models(self):
        """
        A list of available synaptic plasticity models for the current
        configuration (weight dependence, timing dependence, ...) in the
        current simulator.
        """
        pm = self.neuron.possible_models
        for psr in self.post_synaptic_receptors.values():
            pm = pm.intersection(psr.possible_models)
        if len(pm) == 0:
            raise errors.NoModelAvailableError("No possible models for this combination")
        return pm

    @property
    def nest_name(self):
        # todo: make this work with on_grid, off_grid
        available_models = nest.node_models
        suitable_models = self.possible_models.intersection(available_models)
        if len(suitable_models) == 0:
            err_msg = (
                "Model not available in this build of NEST. "
                f"You requested one of: {self.possible_models}"
                f"Available models are: {available_models}"
            )
            raise errors.NoModelAvailableError(err_msg)
        elif len(suitable_models) > 1:
            logger.warning("Several models are available for this set of components")
            logger.warning(", ".join(model for model in suitable_models))
            model = list(suitable_models)[0]
            logger.warning("By default, %s is used" % model)
        else:
            model, = suitable_models  # take the only entry
        return {"on_grid": model,
                "off_grid": model}

    @property
    def variable_map(self):
        var_map = self.neuron.variable_map.copy()
        for name, psr in self.post_synaptic_receptors.items():
            for variable, translated_variable in psr.variable_map.items():
                value = f"{translated_variable}_{self.get_receptor_type(name)}"
                var_map[f"{name}.{variable}"] = value
        return var_map

    @property
    def native_parameters(self):
        """
        A :class:`ParameterSpace` containing parameter names and values
        translated from the standard PyNN names and units to simulator-specific
        ("native") names and units.
        """
        translated_parameters = self.neuron.native_parameters
        # work-in-progress: this assumes all receptors have the same model
        #                   this will not be true in the general case
        # also, not all models with multiple receptor types are "multisynapse" models,
        # e.g. ht_neuron

        # transform list of dicts into dict of lists
        receptor_params = defaultdict(list)
        for name in self.receptor_types:
            psr = self.post_synaptic_receptors[name]
            for name, value in psr.native_parameters.items():
                receptor_params[name].append(value)

        # merge list of lazyarray values into a single lazyarray
        # for now, assume homogeneous parameters
        for name, list_of_values in receptor_params.items():
            ops = [value.operations for value in list_of_values]
            for op in ops[1:]:
                assert op == ops[0]
            if all(value.is_homogeneous for value in list_of_values):
                arrval = ArrayParameter([value.base_value for value in list_of_values])
            else:
                raise NotImplementedError("to do")
            lval = LazyArray(arrval, dtype=ArrayParameter)
            if ops:
                lval.operations = ops[0]
            receptor_params[name] = lval

        translated_parameters.update(**receptor_params)
        return translated_parameters

    def get_native_names(self, *names):
        neuron_names = self.neuron.get_native_names(*[name for name in names if "." not in name])
        for name in names:
            if "." in name:
                psr_name, param_name = name.split(".")
                index = self.receptor_types.index(psr_name)
                tr_name = self.post_synaptic_receptors[psr_name].get_native_names(param_name)
                neuron_names.append("{}[{}]".format(tr_name, index))
        return neuron_names

    def reverse_translate(self, native_parameters):
        standard_parameters = self.neuron.reverse_translate(native_parameters)
        return standard_parameters
