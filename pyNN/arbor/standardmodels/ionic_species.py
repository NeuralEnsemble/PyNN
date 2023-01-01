# ~/arbor/standardmodels/ionic_species.py

from pyNN.parameters import IonicSpecies
from pyNN.models import BaseModelType
from pyNN.standardmodels import StandardModelType
from pyNN.standardmodels import build_translations


class BaseIonicSpeciesModel(BaseModelType):
    """Base class for ionic species models."""
    # Probably should be in ~/pyNN/models.py
    pass


class StandardIonicSpecies(StandardModelType, BaseIonicSpeciesModel):
    """Standard class for ionic species models"""

    # Probably should be in ~/pyNN/standardmodels/__init__.py
    def get_schema(self):
        return {
            # "ion_name": str,
            "reversal_potential": float,
            "internal_concentration": float,
            "external_concentration": float
        }


# Default parameter values from https://github.com/arbor-sim/arbor/blob/master/arbor/cable_cell_param.cpp
class StandardNaIon(StandardIonicSpecies):
    # Probably should be NaIon in ~/pyNN/standardmodels/ionic_species.py
    default_parameters = {
        # "ion_name": "na", triggers pyNN.errors.InvalidParameterValueError
        "reversal_potential": 90.0,  # 115 - 65. mV
        "internal_concentration": 10.0,  # mM
        "external_concentration": 140.0  # mM
    }


class StandardKIon(StandardIonicSpecies):
    # Probably should be KIon in ~/pyNN/standardmodels/ionic_species.py
    default_parameters = {
        # "ion_name": "k", pyNN.errors.InvalidParameterValueError
        "reversal_potential": 26.5,  # -12 - 65. mV
        "internal_concentration": 54.4,  # mM
        "external_concentration": 2.5  # mM
    }


class StandardCaIon(StandardIonicSpecies):
    # Probably should be CaIon in ~/pyNN/standardmodels/ionic_species.py
    default_parameters = {
        # "ion_name": "ca", pyNN.errors.InvalidParameterValueError
        "reversal_potential": 132.5,  # 12.5*std::log(2.0/5e-5) mV
        "internal_concentration": 5e-5,  # mM
        "external_concentration": 2.0  # mM
    }


class NaIon(StandardNaIon):
    """
    Specific ion name class for intuitive use.
    Does not add any other properties or methods to the class.
    """
    translations = build_translations(("reversal_potential", "rev_pot"),  # (pynn_name, sim_name)
                                      ("internal_concentration", "int_con"),
                                      ("external_concentration", "ext_con"), )
    model = "na"


class KIon(StandardKIon):
    """
    Specific ion name class for intuitive use.
    Does not add any other properties or methods to the class.
    """
    translations = build_translations(("reversal_potential", "rev_pot"),  # (pynn_name, sim_name)
                                      ("internal_concentration", "int_con"),
                                      ("external_concentration", "ext_con"), )
    model = "k"


class CaIon(StandardCaIon):
    """
    Specific ion name class for intuitive use.
    Does not add any other properties or methods to the class.
    """
    translations = build_translations(("reversal_potential", "rev_pot"),  # (pynn_name, sim_name)
                                      ("internal_concentration", "int_con"),
                                      ("external_concentration", "ext_con"), )
    model = "ca"
