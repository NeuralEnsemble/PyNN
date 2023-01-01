# encoding: utf-8
"""
Functions to create Arbor decorate ion channels.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
import arbor
from pyNN.arborproto.procedures.arbor_DSL import get_region_DSL


class DecorateIonChannelsMain(object):
    """
    This class is hidden and used below by DecorateIonChannels
    """

    def __init__(self, arbor_labels, ion_channels, other_parameters):
        self._arbor_labels = arbor_labels
        self.ion_channels = ion_channels
        self.other_parameters = other_parameters

    def _decorate_ion_channels(self, parent_decor):
        """
        Call this as self.__decorate_ion_channels() to update self._arbor_labels & self._decor.
        Decoration in this context means painting channel mechanisms on respective
        regions of self._arbor_morphology
        """
        # decor = list_decor_labels[0]
        # arbor_labels = list_decor_labels[1]
        for name, ion_chnl in self.ion_channels.items():
            parameters = self.other_parameters[name]
            region_selector = parameters["conductance_density"].selector
            density_func_name = parameters["conductance_density"].__class__.__name__
            chnl_mech = self.__create_arbor_channel_mechanism(ion_chnl, parameters)
            # get location & append location into self.__arbor_labels
            # create mechanism
            # set decor.pant with arguments location and arbor.density with mechanism as its argument
            # if not isinstance(parameters["conductance_density"].selector, MorphologyFilter):
            #     self._decor.paint(self.__format_extracted_channel_region(parameters), chnl_mech)
            # else:
            #     raise NotImplementedError(density_func_name + ' is not supported')
            #
            parent_decor.paint(self.__paint_region(name, region_selector, density_func_name), chnl_mech)
            # decor.paint('"soma"', density(pas))
            # decor.paint('"soma"',
            #             density('pas', {'g': 0.1}))  # Error: can't place the same mechanism on overlapping regions
            # decor.paint('"soma"', density('pas/e=-45'))
            # return decor, arbor_labels
            return self._arbor_labels, parent_decor

    @staticmethod
    def __create_arbor_channel_mechanism(chnl, its_parameters):
        # Returns arbor's channel mechanisms as a values of keys in the dictionary
        # Dictionary keys corresponds to respective model name associated with an ion channel
        model = chnl.model
        dict_for_chnl = {chnl.translations["conductance_density"]["translated_name"]:
                             its_parameters["conductance_density"].value}
        if model == "pas":
            for ky in its_parameters.keys():
                if ky != "conductance_density":
                    if ky == "e_rev":
                        model = "pas/e=" + str(its_parameters[ky])
                    else:
                        dict_for_chnl.update({chnl.translations[ky]["translated_name"]: its_parameters[ky]})
        else:
            dict_for_chnl.update({"gl": 0.})
            for ky in its_parameters.keys():
                if ky != "conductance_density":
                    dict_for_chnl.update({chnl.translations[ky]["translated_name"]: its_parameters[ky]})
            if model == "na":
                dict_for_chnl.update({"gkbar": 0.})
            else:
                dict_for_chnl.update({"gnabar": 0.})
        if not dict_for_chnl:
            return arbor.density(model)  # arbor.mechanism -> arbor.density
        else:
            return arbor.density(model, dict_for_chnl)  # arbor.mechanism -> arbor.density

    def __paint_region(self, chnl_name, region_selector, density_func_name):
        if density_func_name == "uniform":
            key = "ionchnl_" + chnl_name + "_in_" + region_selector
            # self._arbor_labels.update({key: get_region_DSL(region_selector)})
            self._arbor_labels[key] = get_region_DSL(region_selector)
            return '"{}"'.format(key)
        elif density_func_name == "by_distance":
            raise NotImplementedError(density_func_name + ' is not yet supported')
        elif density_func_name == "by_diameter":
            raise NotImplementedError(density_func_name + ' is not yet supported')
        else:
            raise NotImplementedError(density_func_name + ' is not supported')

    def __create_arbor_channel_location(chnl_parameters):
        # update self.__arbor_labels
        loc_name = chnl_parameters["conductance_density"].selector
        loc_tag = self.__get_swc_tag(loc_name)
        for i in range(self._arbor_morphology.num_branches):
            for seg in self._arbor_morphology.branch_segments(i):
                if seg.tag == loc_tag:
                    # print("this seg")
                    # str(location 1 chnl_parameters["conductance_density"].value_in(self.morphology, loc_tag))
                    pass


class DecorateIonChannels(object):
    """
    Use:
    ```
    from pyNN.arbor.procedures.step6 import DecorateIonChannels
    self._arbor_labels, self._decor = DecorateIonChannels(self._decor, self._arbor_labels, self.ion_channels, other_parameters)
    ```
    """

    def __new__(cls, parent_decor, arbor_labels, ion_channels, other_parameters):
        if len(ion_channels) == 0:
            return arbor_labels, parent_decor
        else:  # creating new instance of DecorateIonChannelsMain from DecorateIonChannels
            class_inst = DecorateIonChannelsMain(arbor_labels, ion_channels, other_parameters)
            return class_inst._decorate_ion_channels(parent_decor)
