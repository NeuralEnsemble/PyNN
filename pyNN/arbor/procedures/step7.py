# encoding: utf-8
"""
Functions to create Arbor decorate synapse.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import arbor
# from pyNN.arbor.procedures.swc_tags_names import get_swc_tag
from pyNN.arbor.procedures.arbor_DSL import get_synaptic_locset_DSL_uniformly_per_um, \
    get_synaptic_locset_DSL_distance_per_um


class DecorateSynapseMain(object):
    """
    This class is hidden and used below by DecorateSynapse
    """

    def __init__(self, arbor_labels, post_synaptic_entities, other_parameters):
        self._arbor_labels = arbor_labels
        self.post_synaptic_entities = post_synaptic_entities
        self.other_parameters = other_parameters

    def _decorate_synaptic_entities(self, parent_decor, arbor_morphology):
        """
        Call this as self.__decorate_synaptic_entities(arbor_morphology) to update self._arbor_labels & self._decor.
        Decoration in this context means painting channel mechanisms on respective
        regions of self._arbor_morphology
        """
        for name, pse in self.post_synaptic_entities.items():
            parameters = self.other_parameters[name]
            syn_mech = self.__create_arbor_syn_mechanism(pse, parameters)
            parent_decor.place(self.__place_loc(name, parameters, arbor_morphology), syn_mech)
            return self._arbor_labels, parent_decor

    @staticmethod
    def __create_arbor_syn_mechanism(pse, its_parameters):
        # Returns arbor's channel mechanisms as a values of keys in the dictionary
        # Dictionary keys corresponds to respective model name associated with an ion channel
        model = pse.model
        # dict_for_pse = {pse.translations["density"]["translated_name"]: its_parameters["density"].value}
        dict_for_pse = {}
        if model == "expsyn":
            for ky in its_parameters.keys():
                if ky != "density":
                    dict_for_pse.update({pse.translations[ky]["translated_name"]: its_parameters[ky]})
        # else other cases
        if dict_for_pse == {}:
            return arbor.synapse(model)
        else:
            return arbor.synapse(model, dict_for_pse)

    def __place_loc(self, syn_name, synparameters, arbormorph):
        density_func = synparameters["density"]
        density_func_name = synparameters["density"].__class__.__name__
        if density_func_name == "uniform":
            loc_selector = synparameters["density"].selector
            um = synparameters["density"].value
            key = "syn_" + syn_name + "_in_" + loc_selector
            val = get_synaptic_locset_DSL_uniformly_per_um(loc_selector, um, arbormorph)
        elif density_func_name == "by_distance":
            loc_selector = density_func.selector.__class__.__name__
            key = "syn_" + syn_name + "_in_" + loc_selector
            val = get_synaptic_locset_DSL_distance_per_um(density_func)
        elif density_func_name == "by_diameter":
            raise NotImplementedError(density_func_name + ' is not yet supported')
        else:
            raise NotImplementedError(density_func_name + ' is not supported')
        self._arbor_labels[key] = val
        return '"{}"'.format(key)


class DecorateSynapse(object):
    """
    Use:
    ```
    from pyNN.arbor.procedures.step7 import DecorateSynapse
    self._arbor_labels, self._decor = DecorateSynapse(self._decor, self._arbor_labels, self.post_synaptic_entities,
                                                     other_parameters, self._arbor_morphology)
    ```
    """

    def __new__(cls, parent_decor, arbor_labels, post_synaptic_entities, other_parameters, arbor_morphology):
        if len(post_synaptic_entities) == 0:
            return arbor_labels, parent_decor
        else:  # creating new instance of DecorateSynapseMain from DecorateSynapse
            class_inst = DecorateSynapseMain(arbor_labels, post_synaptic_entities, other_parameters)
            return class_inst._decorate_synaptic_entities(parent_decor, arbor_morphology)
