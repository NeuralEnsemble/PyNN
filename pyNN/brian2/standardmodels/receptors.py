"""

"""


from copy import deepcopy
import brian2
from brian2 import mV, ms, nF, nA, uS, Hz, nS
from pyNN.standardmodels import receptors, build_translations
from pyNN.parameters import ParameterSpace


conductance_based_exponential_synapses = brian2.Equations('''
    dge/dt = -ge/tau_syn_e  : siemens
    dgi/dt = -gi/tau_syn_i  : siemens
    i_syn = ge*(e_rev_e - v) + gi*(e_rev_i - v)  : amp
    tau_syn_e               : second
    tau_syn_i               : second
    e_rev_e                 : volt
    e_rev_i                 : volt
''')

conductance_based_alpha_synapses = brian2.Equations('''
    dge/dt = (2.7182818284590451*ye-ge)/tau_syn_e  : siemens
    dye/dt = -ye/tau_syn_e                         : siemens
    dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_i  : siemens
    dyi/dt = -yi/tau_syn_i                         : siemens
    i_syn = ge*(e_rev_e - v) + gi*(e_rev_i - v)    : amp
    tau_syn_e               : second
    tau_syn_i               : second
    e_rev_e                 : volt
    e_rev_i                 : volt
''')

current_based_exponential_synapses = brian2.Equations('''
    die/dt = -ie/tau_syn_e  : amp
    dii/dt = -ii/tau_syn_i  : amp
    i_syn = ie + ii         : amp
    tau_syn_e               : second
    tau_syn_i               : second
''')

current_based_alpha_synapses = brian2.Equations('''
    die/dt = (2.7182818284590451*ye-ie)/tau_syn_e : amp
    dye/dt = -ye/tau_syn_e                        : amp
    dii/dt = (2.7182818284590451*yi-ii)/tau_syn_e : amp
    dyi/dt = -yi/tau_syn_e                        : amp
    i_syn = ie + ii                               : amp
    tau_syn_e                                     : second
    tau_syn_i                                     : second
''')

conductance_based_synapse_translations = build_translations(
                ('tau_syn_E',  'tau_syn_e',  lambda **p: p["tau_syn_E"] * ms, lambda **p: p["tau_syn_e"] / ms),
                ('tau_syn_I',  'tau_syn_i',  lambda **p: p["tau_syn_I"] * ms, lambda **p: p["tau_syn_i"] / ms),
                ('e_rev_E',    'e_rev_e',    lambda **p: p["e_rev_E"] * mV, lambda **p: p["e_rev_e"] / mV),
                ('e_rev_I',    'e_rev_i',    lambda **p: p["e_rev_I"] * mV, lambda **p: p["e_rev_i"] / mV))

current_based_synapse_translations = build_translations(
                ('tau_syn_E',  'tau_syn_e',  lambda **p: p["tau_syn_E"] * ms, lambda **p: p["tau_syn_e"] / ms),
                ('tau_syn_I',  'tau_syn_i',  lambda **p: p["tau_syn_I"] * ms, lambda **p: p["tau_syn_i"] / ms))

conductance_based_variable_translations = build_translations(
                ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
                ('gsyn_exc', 'ge', lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_inh', 'gi', lambda p: p * uS, lambda p: p/ uS))

current_based_variable_translations = build_translations(
                ('v',         'v',         lambda p: p * mV, lambda p: p/ mV), #### change p by p["v"]
                ('isyn_exc', 'ie',         lambda p: p * nA, lambda p: p/ nA),
                ('isyn_inh', 'ii',         lambda p: p * nA, lambda p: p/ nA))


class PSRMixin:

    def native_parameters(self, suffix):
        return self.translate(self.parameter_space, suffix)

    def computed_parameters(self):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return []  # hardcoded as temporary hack

    def get_native_names(self, *names, suffix=None):
        """
        Return a list of native parameter names for a given model.
        """
        if names:
            translations = (self.translations(suffix)[name] for name in names)
        else:  # return all names
            translations = self.translations(suffix).values()
        return [D['translated_name'] for D in translations]

    def translate(self, parameters, suffix, copy=True):
        """Translate standardized model parameters to simulator-specific parameters."""
        if copy:
            _parameters = deepcopy(parameters)
        else:
            _parameters = parameters
        cls = self.__class__
        if parameters.schema != self.get_schema():
            raise Exception("Schemas do not match: %s != %s" % (parameters.schema, self.get_schema()))  # should replace this with a PyNN-specific exception type
        native_parameters = {}
        for name in parameters.keys():
            D = self.translations(suffix)[name]
            pname = D['translated_name']
            if callable(D['forward_transform']):
                pval = D['forward_transform'](**_parameters)
            else:
                try:
                    pval = eval(D['forward_transform'], globals(), _parameters)
                except NameError as errmsg:
                    raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s"
                                    % (pname, cls.__name__, D['forward_transform'], parameters, errmsg))
                except ZeroDivisionError:
                    raise
                    #pval = 1e30 # this is about the highest value hoc can deal with
            native_parameters[pname] = pval
        return ParameterSpace(native_parameters, schema=None, shape=parameters.shape)

    def reverse_translate(self, native_parameters, suffix):
        """Translate simulator-specific model parameters to standardized parameters."""
        cls = self.__class__
        standard_parameters = {}
        for name, D in self.translations(suffix).items():
            tname = D['translated_name']
            if tname in native_parameters.keys():
                if callable(D['reverse_transform']):
                    standard_parameters[name] = D['reverse_transform'](**native_parameters)
                else:
                    try:
                        standard_parameters[name] = eval(D['reverse_transform'], {}, native_parameters)
                    except NameError as errmsg:
                        raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s"
                                        % (name, cls.__name__, D['reverse_transform'], native_parameters, errmsg))
        return ParameterSpace(standard_parameters, schema=self.get_schema(), shape=native_parameters.shape)


class CondExpPostSynapticResponse(PSRMixin, receptors.CondExpPostSynapticResponse):

    recordable = ["gsyn"]

    def eqs(self, suffix):
        return  brian2.Equations(f'''
            dg{suffix}/dt = -g{suffix}/tau_syn_{suffix}  : siemens
            tau_syn_{suffix}  : second
            e_rev_{suffix}  : volt
        ''')

    def translations(self, suffix):
        return build_translations(
            (f'tau_syn',  f'tau_syn_{suffix}',  lambda **p: p[f"tau_syn"] * ms, lambda **p: p[f"tau_syn_{suffix}"] / ms),
            (f'e_syn',    f'e_rev_{suffix}',    lambda **p: p[f"e_syn"] * mV, lambda **p: p[f"e_rev_{suffix}"] / mV),
        )

    def state_variable_translations(self, suffix):
        return build_translations(
            (f'{suffix}.gsyn', f'g{suffix}', lambda p: p * uS, lambda p: p/ uS),
        )

    def post_synaptic_variable(self, suffix):
        return f"g{suffix}"

    def synaptic_current(self, suffix):
        return f"g{suffix}*(e_rev_{suffix} - v)"


class CondAlphaPostSynapticResponse(PSRMixin, receptors.CondAlphaPostSynapticResponse):

    recordable = ["gsyn", "ysyn"]

    def eqs(self, suffix):
        return  brian2.Equations(f'''
            dg{suffix}/dt = (2.7182818284590451 * y{suffix} - g{suffix}) / tau_syn_{suffix}  : siemens
            dy{suffix}/dt = -y{suffix}/tau_syn_{suffix}  : siemens
            tau_syn_{suffix}  : second
            e_rev_{suffix}  : volt
        ''')

    def translations(self, suffix):
        return build_translations(
            (f'tau_syn',  f'tau_syn_{suffix}',  lambda **p: p[f"tau_syn"] * ms, lambda **p: p[f"tau_syn_{suffix}"] / ms),
            (f'e_syn',    f'e_rev_{suffix}',    lambda **p: p[f"e_syn"] * mV, lambda **p: p[f"e_rev_{suffix}"] / mV),
        )

    def state_variable_translations(self, suffix):
        return build_translations(
            (f'{suffix}.gsyn', f'g{suffix}', lambda p: p * uS, lambda p: p/ uS),
            (f'{suffix}.ysyn', f'y{suffix}', lambda p: p * uS, lambda p: p/ uS),
        )

    def post_synaptic_variable(self, suffix):
        return f"y{suffix}"

    def synaptic_current(self, suffix):
        return f"g{suffix}*(e_rev_{suffix} - v)"


AlphaPSR = CondAlphaPostSynapticResponse  # alias
ExpPSR = CondExpPostSynapticResponse  # alias