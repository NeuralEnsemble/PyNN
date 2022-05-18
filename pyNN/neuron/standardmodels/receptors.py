from neuron import h
from pyNN.standardmodels import receptors, build_translations


class CondAlphaPostSynapticResponse(receptors.CondAlphaPostSynapticResponse):

    translations = build_translations(
        ('e_syn', 'e'),
        ('tau_syn', 'tau')
    )
    model = h.AlphaSyn

AlphaPSR = CondAlphaPostSynapticResponse  # alias