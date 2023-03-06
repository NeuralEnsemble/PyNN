"""
Definition of default parameters (and hence, standard parameter names) for
standard post-synaptic response models.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""


from .base import StandardPostSynapticResponse


class CurrExpPostSynapticResponse(StandardPostSynapticResponse):
    """
    Post-synaptic response consisting of a step increase in synaptic current
    followed by exponential decay.
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
    Post-synaptic response consisting of a step increase in synaptic conductance
    followed by exponential decay.
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
    Post-synaptic response consisting of an "alpha-function"-shaped synaptic conductance:

        g(t) = t * exp(1 - t/tau_syn)

    (see A. Roth and M. C. W. van Rossum (2013) Modeling Synapses.
     In: Computational Modeling Methods for Neuroscientists, MIT Press 2013, pp 139-160)
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
    Post-synaptic response consisting of an beta-function-shaped synaptic conductance.
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
