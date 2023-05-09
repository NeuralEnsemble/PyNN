"""
Cell models generated from 9ML

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from itertools import chain

logger = logging.getLogger("PyNN")


# Neuron Models derived from a 9ML AL definition

# class NineMLCellType(BaseCellType):
#    #model = NineMLCell
#
#    def __init__(self, parameters):
#        BaseCellType.__init__(self, parameters)
#        self.parameters["type"] = self


# def unimplemented_builder(*args, **kwargs):
#    raise NotImplementedError("TODO: 9ML neuron builder")


# def nineml_cell_type(name, neuron_model, port_map={}, weight_variables={}, **synapse_models):
#    """
#    Return a new NineMLCellType subclass.
#    """
#    return _build_nineml_celltype(name, (NineMLCellType,),
#                                  {'neuron_model': neuron_model,
#                                   'synapse_models': synapse_models,
#                                   'port_map': port_map,
#                                   'weight_variables': weight_variables,
#                                   'builder': unimplemented_builder})


# Helpers for Neuron Models derived from a 9ML AL definition

def _add_prefix(synapse_model, prefix, port_map):
    assert False, "Deprecated"
    """
    Add a prefix to all variables in `synapse_model`, except for variables with
    receive ports and specified in `port_map`.
    """
    synapse_model.__cache__ = {}
    exclude = []
    new_port_map = []
    for name1, name2 in port_map:
        if synapse_model.ports_map[name2].mode == 'recv':
            exclude.append(name2)
            new_port_map.append((name1, name2))
        else:
            new_port_map.append((name1, prefix + '_' + name2))
    synapse_model.add_prefix(prefix + '_', exclude=exclude)
    return new_port_map


_default_units = {
    "time": "ms",
    "voltage": "mV",
    "current": "nA"
}


class build_nineml_celltype(type):
    """
    Metaclass for building NineMLCellType subclasses
    Called by nineml_celltype_from_model
    """
    def __new__(cls, name, bases, dct):

        import nineml.abstraction as al
        from nineml.abstraction.dynamics.utils import (
            flattener, modifiers)

        # Extract Parameters Back out from Dict:
        combined_model = dct['combined_model']
        weight_vars = dct['weight_variables']

        # Flatten the model:
        assert isinstance(combined_model, al.ComponentClass)
        if combined_model.is_flat():
            flat_component = combined_model
        else:
            flat_component = flattener.flatten(combined_model, name)

        # Make the substitutions:
        flat_component.backsub_all()
        # flat_component.backsub_aliases()
        # flat_component.backsub_equations()

        # Close any open reduce ports:
        modifiers.DynamicPortModifier.close_all_reduce_ports(componentclass=flat_component)

        # New:
        dct["combined_model"] = flat_component
        dct["default_parameters"] = dict((param.name, 1.0) for param in flat_component.parameters)
        dct["default_initial_values"] = dict((statevar.name, 0.0)
                                             for statevar in chain(flat_component.state_variables))
        dct["receptor_types"] = list(weight_vars.keys())
        dct["standard_receptor_type"] = (dct["receptor_types"] == ('excitatory', 'inhibitory'))
        # how to determine this?
        # neuron component has a receive analog port with dimension current,
        # that is not connected to a synapse port?
        dct["injectable"] = False
        # how to determine this?
        # synapse component has a receive analog port with dimension voltage?
        dct["conductance_based"] = True
        dct["model_name"] = name
        dct["units"] = dict((statevar.name, _default_units[statevar.dimension.name])
                            for statevar in chain(flat_component.state_variables))

        # Recording from bindings:
        dct["recordable"] = ([port.name for port in flat_component.analog_ports]
                             + ['spikes', 'regime']
                             + [alias.lhs for alias in flat_component.aliases]
                             + [statevar.name for statevar in flat_component.state_variables])

        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))
        dct["builder"](flat_component, dct["weight_variables"], hierarchical_mode=True)

        return type.__new__(cls, name, bases, dct)
