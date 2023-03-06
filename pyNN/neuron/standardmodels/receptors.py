from neuron import h
from pyNN.standardmodels import receptors, build_translations


class CurrExpPostSynapticResponse(receptors.CurrExpPostSynapticResponse):

    translations = build_translations(
        ('tau_syn', 'tau')
    )
    model = h.ExpISyn
    recordable = ["isyn"]
    variable_map = {"isyn": "i"}


class CondExpPostSynapticResponse(receptors.CondExpPostSynapticResponse):

    translations = build_translations(
        ('e_syn', 'e'),
        ('tau_syn', 'tau')
    )
    model = h.ExpSyn
    recordable = ["gsyn"]
    variable_map = {"gsyn": "g"}


class CondAlphaPostSynapticResponse(receptors.CondAlphaPostSynapticResponse):

    translations = build_translations(
        ('e_syn', 'e'),
        ('tau_syn', 'tau')
    )
    model = h.AlphaSyn
    recordable = ["gsyn"]
    variable_map = {"gsyn": "g"}


# create shorter aliases

AlphaPSR = CondAlphaPostSynapticResponse
ExpPSR = CondExpPostSynapticResponse
