from pyNN.standardmodels import receptors, build_translations


class CondAlphaPostSynapticResponse(receptors.CondAlphaPostSynapticResponse):
    possible_models = set(["aeif_cond_alpha_multisynapse"])

    translations = build_translations(
        ('e_syn', 'E_rev'),
        ('tau_syn', 'tau_syn')
    )

class CondBetaPostSynapticResponse(receptors.CondBetaPostSynapticResponse):
    possible_models = set(["aeif_cond_beta_multisynapse"])

    translations = build_translations(
        ('e_syn', 'E_rev'),
        ('tau_rise', 'tau_rise'),
        ('tau_decay', 'tau_decay')
    )

AlphaPSR = CondAlphaPostSynapticResponse  # alias
BetaPSR = CondBetaPostSynapticResponse  # alias
