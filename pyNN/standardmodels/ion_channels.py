"""



"""


from pyNN.standardmodels import StandardIonChannelModel


class NaChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.12,
        "e_rev": 50.0
    }


class KdrChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.036,
        "e_rev": -77.0
    }


class PassiveLeak(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.0003,
        "e_rev": -65.0
    }
