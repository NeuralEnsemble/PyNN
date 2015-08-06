"""
Definition of NativeCellType class for NEST.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import warnings
import nest
from pyNN.models import BaseCellType
from pyNN.parameters import Sequence
from . import conversion

UNITS_MAP = {
    'spikes': 'ms',
    'V_m': 'mV',
}


def get_defaults(model_name):
    valid_types = (int, float, Sequence)
    defaults = nest.GetDefaults(model_name)
    variables = defaults.get('recordables', [])
    ignore = ['archiver_length', 'available', 'capacity', 'elementsize',
              'frozen', 'instantiations', 'local', 'model', 'recordables',
              'state', 't_spike', 'tau_minus', 'tau_minus_triplet',
              'thread', 'vp', 'receptor_types', 'events', 'global_id',
              'element_type', 'type', 'type_id', 'has_connections', 'n_synapses',
              'thread_local_id']
    default_params = {}
    default_initial_values = {}
    for name, value in defaults.items():
        if name in variables:
            default_initial_values[name] = value
        elif name not in ignore:
            if isinstance(value, valid_types):
                default_params[name] = conversion.make_pynn_compatible(value)
            else:
                warnings.warn("Ignoring parameter '%s' since PyNN does not support %s" % (name, type(value)))
    return default_params, default_initial_values


def get_receptor_types(model_name):
    return list(nest.GetDefaults(model_name).get("receptor_types", ('excitatory', 'inhibitory')))


def get_recordables(model_name):
    return [sl.name for sl in nest.GetDefaults(model_name).get("recordables", [])]


def native_cell_type(model_name):
    """
    Return a new NativeCellType subclass.
    """
    assert isinstance(model_name, str)
    default_parameters, default_initial_values = get_defaults(model_name)
    receptor_types = get_receptor_types(model_name)
    recordable = get_recordables(model_name) + ['spikes']
    element_type = nest.GetDefaults(model_name, 'element_type').name
    return type(model_name,
                (NativeCellType,),
                {'nest_model': model_name,
                 'default_parameters': default_parameters,
                 'default_initial_values': default_initial_values,
                 'receptor_types': receptor_types,
                 'injectable': ("V_m" in default_initial_values),
                 'recordable': recordable,
                 'units': dict(((var, UNITS_MAP.get(var, 'unknown')) for var in recordable)),
                 'standard_receptor_type': (receptor_types == ['excitatory', 'inhibitory']),
                 'nest_name': {"on_grid": model_name, "off_grid": model_name},
                 'conductance_based': ("g" in (s[0] for s in recordable)),
                 'always_local': element_type == 'stimulator',
                 'uses_parrot': element_type == 'stimulator'
                 })


class NativeCellType(BaseCellType):

    def get_receptor_type(self, name):
        return nest.GetDefaults(self.nest_model)["receptor_types"][name]
