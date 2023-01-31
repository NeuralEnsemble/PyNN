"""


"""

from pyNN.standardmodels import StandardPostSynapticResponse


class CurrExpPostSynapticResponse(StandardPostSynapticResponse):
    """

    """

    default_parameters = {
        'tau_syn': 5.0  # time constant of the synaptic conductance in ms.
    }
    default_initial_values = {
        "isyn": 0.0
    }
    units = {
        "isyn": "nA"
    }
    conductance_based = False


class CondExpPostSynapticResponse(StandardPostSynapticResponse):
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
    conductance_based = True


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
    conductance_based = True


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
    conductance_based = True
