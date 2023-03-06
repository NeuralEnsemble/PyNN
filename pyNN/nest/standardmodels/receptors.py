from ...standardmodels import receptors, build_translations


class CurrExpPostSynapticResponse(receptors.CurrExpPostSynapticResponse):
    possible_models = set(["iaf_psc_exp_multisynapse"])

    translations = build_translations(
        ('tau_syn', 'tau_syn')
    )
    recordable = ["isyn"]
    scale_factors = {"isyn": 0.001}
    variable_map = {"isyn": "I_syn"}


class CondExpPostSynapticResponse(receptors.CondExpPostSynapticResponse):
    possible_models = set(["gif_cond_exp_multisynapse"])

    translations = build_translations(
        ('e_syn', 'E_rev'),
        ('tau_syn', 'tau_syn')
    )
    recordable = ["gsyn"]
    scale_factors = {"gsyn": 0.001}
    variable_map = {"gsyn": "g"}


class CondAlphaPostSynapticResponse(receptors.CondAlphaPostSynapticResponse):
    possible_models = set(["aeif_cond_alpha_multisynapse"])

    translations = build_translations(
        ('e_syn', 'E_rev'),
        ('tau_syn', 'tau_syn')
    )
    recordable = ["gsyn"]
    scale_factors = {"gsyn": 0.001}
    variable_map = {"gsyn": "g"}


class CondBetaPostSynapticResponse(receptors.CondBetaPostSynapticResponse):
    possible_models = set(["aeif_cond_beta_multisynapse"])

    translations = build_translations(
        ('e_syn', 'E_rev'),
        ('tau_rise', 'tau_rise'),
        ('tau_decay', 'tau_decay')
    )
    recordable = ["gsyn"]
    scale_factors = {"gsyn": 0.001}
    variable_map = {"gsyn": "g"}


# create shorter aliases

ExpPSR = CondExpPostSynapticResponse
AlphaPSR = CondAlphaPostSynapticResponse
BetaPSR = CondBetaPostSynapticResponse
