# encoding: utf-8
"""
Functions to create Arbor decorate ion species.

:copyright: Copyright 2006-2022 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""


class DecorateIonicSpeciesMain(object):
    """
    This class is hidden and used below by DecorateIonicSpecies
    """

    def __init__(self, ionic_species, other_parameters):
        self.ionic_species = ionic_species
        self.other_parameters = other_parameters

    def _decorate_ionic_species(self, parent_decor):
        """
        Call this as self.__decorate_ionic_species(other_parameters) to update self._decor
        """
        # list_of_tuples = []
        for name, ion_species in self.ionic_species.items():
            parameters = self.other_parameters[name]
            dict_input = self.__get_dict_for_ion(ion_species, parameters)
            parent_decor.set_ion(ion_species.model, **dict_input)
            # list_of_tuples.append((ion_species.model, dict_input))
        return parent_decor  # return list_of_tuples

    @staticmethod
    def __get_dict_for_ion(ion_species, parameters):
        dict_for_ion = {}
        # dicts = {"int_con": 5.0, "rev_pot": 70, "method": None}
        # decor.set_ion("na", **dicts)
        for ky in parameters.keys():
            if ky == "method":
                # dict_for_ion.update({"method": parameters[ky]})
                dict_for_ion.update({"method": "nernst/" + ion_species.model})
            else:
                dict_for_ion.update({ion_species.translations[ky]["translated_name"]: parameters[ky]})
        return dict_for_ion


class DecorateIonicSpecies(object):
    """
    Use:
    ```
    from pyNN.arbor.procedures.step5 import DecorateIonicSpecies
    self._decor = DecorateIonicSpecies(self._decor, self.ionic_species, other_parameters)
    ```
    """

    def __new__(cls, parent_decor, ionic_species, other_parameters):
        #  creating new instance of DecorateIonicSpeciesMain from DecorateIonicSpecies
        class_inst = DecorateIonicSpeciesMain(ionic_species, other_parameters)
        return class_inst._decorate_ionic_species(parent_decor)
