"""



"""


from pyNN.standardmodels import StandardIonChannelModel, StandardPostSynapticResponseModel
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


class CondExpPostSynapticResponse(StandardPostSynapticResponseModel):
    default_parameters = {
        "density": uniform('all', 0.5),  # synapses per micron
        "e_rev": 0.0,
        "tau_syn": 5.0
    }