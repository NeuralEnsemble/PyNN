"""


"""

from pyNN.standardmodels import StandardPostSynapticResponse


class CondAlphaPostSynapticResponse(StandardPostSynapticResponse):
    """
    
    """

    default_parameters = {
        'e_syn': 0.0,   # synaptic reversal potential in mV.
        'tau_syn': 5.0  # time constant of the synaptic conductance in ms.
    }
    default_initial_values = {
        "gsyn": 0.0
    }
    units = {
        "gsyn": "uS"
    }


class CondBetaPostSynapticResponse(StandardPostSynapticResponse):
    """
    
    """

    default_parameters = {
        'e_syn': 0.0,   # synaptic reversal potential in mV.
        'tau_rise': 0.2,  # rise time constant of the synaptic conductance in ms.
        'tau_decay': 1.7  # decay time constant of the synaptic conductance in ms.
    }
    default_initial_values = {
        "gsyn": 0.0
    }
    units = {
        "gsyn": "uS"
    }
