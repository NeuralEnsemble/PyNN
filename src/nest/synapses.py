"""
Definition of NativeSynapseType class for NEST

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nest
from pyNN.models import BaseModelType, BaseSynapseType

DEFAULT_TAU_MINUS = 20.0


def get_synapse_defaults(model_name):
    defaults = nest.GetDefaults(model_name)
    ignore = ['max_delay', 'min_delay', 'num_connections',
              'num_connectors', 'receptor_type', 'synapsemodel',
              'property_object', 'type']
    default_params = {}
    for name,value in defaults.items():
        if name not in ignore:
            default_params[name] = value
    default_params['tau_minus'] = DEFAULT_TAU_MINUS
    return default_params


def native_synapse_type(model_name):
    """
    Return a new NativeSynapseType subclass.
    """
    assert isinstance(model_name, str)
    default_parameters = get_synapse_defaults(model_name)
    return type(model_name,
                (NativeSynapseType,),
                {
                    'nest_name': model_name,
                    'default_parameters': default_parameters,
                })

class NativeSynapseType(BaseSynapseType):

    @property
    def native_parameters(self):
        return self.parameter_space

    def _get_nest_synapse_model(self, suffix):
        synapse_defaults = {}
        for name, value in self.native_parameters.items():
            if value.is_homogeneous:
                value.shape = (1,)
                synapse_defaults[name] = value.evaluate(simplify=True)
        synapse_defaults.pop("tau_minus")
        label = "%s_%s" % (self.nest_name, suffix)
        nest.CopyModel(self.nest_name, label, synapse_defaults)
        return label

    def get_native_names(self, *names):
        return names

    #def _set_tau_minus(self, cells):
    #    if len(cells) > 0:
    #        if 'tau_minus' in nest.GetStatus([cells[0]])[0]:
    #            self.mechanism.parameter_space["tau_minus"].shape = (1,) # temporary hack
    #            tau_minus = self.mechanism.parameter_space["tau_minus"].evaluate(simplify=True)
    #            nest.SetStatus(cells.tolist(), [{'tau_minus': tau_minus}])
    #        else:
    #            raise Exception("Postsynaptic cell model %s does not support STDP."
    #                            % nest.GetStatus([cells[0]], "model"))
