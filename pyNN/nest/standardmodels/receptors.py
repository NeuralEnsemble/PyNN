from pyNN.standardmodels import receptors, build_translations


class CondAlphaPostSynapticResponse(receptors.CondAlphaPostSynapticResponse):
    possible_models = set(["aeif_cond_alpha_multisynapse"])

    translations = build_translations(
        ('e_syn', 'E_rev'),
        ('tau_syn', 'tau_syn')
    )

AlphaPSR = CondAlphaPostSynapticResponse  # alias