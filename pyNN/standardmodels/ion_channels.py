"""



"""


from pyNN.standardmodels import StandardIonChannelModel


class NaChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.0,
        "e_rev": 40.0
    }


class KdrChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.0,
        "e_rev": -90.0
    }


class PassiveLeak(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.0,
        "e_rev": -65.0
    }
