"""



"""


from pyNN.standardmodels import StandardIonChannelModel
#from pyNN.morphology import uniform


class NaChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.12, #uniform('all', 0.12),
        #"e_rev": 50.0
    }
    default_initial_values = {
#        "m": 0.0,  # todo: make these functions
#        "h": 1.0
    }

class KdrChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.036, #uniform('all', 0.036),
        #"e_rev": -77.0
    }
    default_initial_values = {
#        "n": 1.0
    }


class PassiveLeak(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.0003, #uniform('all', 0.0003),
        "e_rev": -65.0
    }


class PassiveLeakHH(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.0003, #uniform('all', 0.0003),
        "e_rev": -54.3
    }