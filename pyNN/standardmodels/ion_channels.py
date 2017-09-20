"""



"""


from pyNN.standardmodels import StandardIonChannelModel
from pyNN.morphology import uniform


class NaChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": uniform('all', 0.12),
        "e_rev": 50.0
    }


class KdrChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": uniform('all', 0.036),
        "e_rev": -77.0
    }


class PassiveLeak(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": uniform('all', 0.0003),
        "e_rev": -65.0
    }
