"""
Definition of NativeSynapseType class for NEST

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nest
from pyNN.models import BaseModelType, BaseSynapseDynamics

DEFAULT_TAU_MINUS = 20.0

def get_defaults(model_name):
    defaults = nest.GetDefaults(model_name)
    ignore = ['delay', 'max_delay', 'min_delay', 'num_connections',
              'num_connectors', 'receptor_type', 'synapsemodel', 'weight',
              'property_object']
    default_params = {}
    for name,value in defaults.items():
        if name not in ignore:
            default_params[name] = value
    default_params['tau_minus'] = DEFAULT_TAU_MINUS
    return default_params




class NativeSynapseDynamics(BaseSynapseDynamics):

    def __init__(self, model_name, parameters={}):
        cls = type(model_name, (NativeSynapseMechanism,),
                   {'nest_model': model_name})
        self.mechanism = cls(parameters)

    def _get_nest_synapse_model(self, suffix):
        defaults = self.mechanism.parameters.copy()
        defaults.pop("tau_minus")
        label = "%s_%s" % (self.mechanism.nest_model, suffix)
        nest.CopyModel(self.mechanism.nest_model,
                       label,
                       defaults)
        return label

    def _set_tau_minus(self, cells):
        if len(cells) > 0:
            if 'tau_minus' in nest.GetStatus([cells[0]])[0]:
                tau_minus = self.mechanism.parameters["tau_minus"]
                nest.SetStatus(cells.tolist(), [{'tau_minus': tau_minus}])
            else:
                raise Exception("Postsynaptic cell model %s does not support STDP."
                                % nest.GetStatus([cells[0]], "model"))


class NativeSynapseMechanism(BaseModelType):

    def __new__(cls, parameters):
        cls.default_parameters = get_defaults(cls.nest_model)
        return super(NativeSynapseMechanism, cls).__new__(cls)
