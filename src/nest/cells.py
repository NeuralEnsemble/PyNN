"""
Definition of NativeCellType class for NEST.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nest
from pyNN.models import BaseCellType

def get_defaults(model_name):
    defaults = nest.GetDefaults(model_name)
    variables = defaults.get('recordables', [])
    ignore = ['archiver_length', 'available', 'capacity', 'elementsize',
              'frozen', 'instantiations', 'local', 'model', 'recordables',
              'state', 't_spike', 'tau_minus', 'tau_minus_triplet',
              'thread', 'vp', 'receptor_types', 'events']
    default_params = {}
    default_initial_values = {}
    for name,value in defaults.items():
        if name in variables:
            default_initial_values[name] = value
        elif name not in ignore:
            default_params[name] = value
    return default_params, default_initial_values

def get_receptor_types(model_name):
    return nest.GetDefaults(model_name).get("receptor_types", ('excitatory', 'inhibitory'))

def get_recordables(model_name):
    return nest.GetDefaults(model_name).get("recordables", [])

def native_cell_type(model_name):
    """
    Return a new NativeCellType subclass.
    """
    assert isinstance(model_name, str)
    return type(model_name, (NativeCellType,), {'nest_model' : model_name})
    

class NativeCellType(BaseCellType):

    def __new__(cls, parameters):
        cls.default_parameters, cls.default_initial_values = get_defaults(cls.nest_model)
        cls.synapse_types = get_receptor_types(cls.nest_model)
        cls.injectable = ("V_m" in cls.default_initial_values)
        cls.recordable = get_recordables(cls.nest_model) + ['spikes']
        cls.standard_receptor_type = (cls.synapse_types == ('excitatory', 'inhibitory'))
        cls.nest_name  = {"on_grid": cls.nest_model, "off_grid": cls.nest_model}
        cls.conductance_based = ("g" in (s[0] for s in cls.recordable))
        return super(NativeCellType, cls).__new__(cls)

    def get_receptor_type(self, name):
        return nest.GetDefaults(self.nest_model)["receptor_types"][name]
