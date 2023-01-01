# ~/mc/pyNN/arbor/standardmodels/ion_channels.py
"""
"""

from pyNN.standardmodels import ion_channels as standard, build_translations


class NaChannel(standard.NaChannel):
    """
    Exactly the same as neuron.NaChannel because
    https://arbor.readthedocs.io/en/latest/concepts/mechanisms.html#density-mechanisms
    https://arbor.readthedocs.io/en/latest/internals/nmodl.html
    """
    translations = build_translations(
        ("conductance_density", "gnabar"), #(pynn_name, sim_name)
    )
    #variable_translations = {"m": ("hh", "m"),
    #                         "h": ("hh", "h"),}
    model = "hh"
    ion_name = "na" #conductance_density_parameter = "gnabar"
    
class KdrChannel(standard.KdrChannel):
    translations = build_translations(
        ('conductance_density', 'gkbar'),
    )
    #variable_translations = {'n': ('hh', 'n')}
    model = "hh"
    ion_name = "k" #conductance_density_parameter = 'gkbar'


class PassiveLeak(standard.PassiveLeak):
    translations = build_translations(
        ('conductance_density', 'g'),
        ('e_rev', 'e'),
    )
    model = "pas" #conductance_density_parameter = 'g'


class CondExpPostSynapticResponse(standard.CondExpPostSynapticResponse):
    """
    Synapse with discontinuous change in conductance at an event followed by an exponential decay.
    """
    translations = build_translations(
        ('density', 'density'),
        ('e_rev', 'e'),
        ('tau_syn', 'tau')
    )
    model = "expsyn"
    
class Cond2ExpPostSynapticResponse(standard.CondExpPostSynapticResponse):
    """
    Bi-exponential conductance synapse described by two time constants: rise and decay
    """
    default_parameters = {} # standard.Cond2ExpPostSynapticResponse does NOT exist
    translations = build_translations(
        ('density', 'density'),
        ('e_rev', 'e'),
        #('tau_syn', 'tau1') ALSO ('tau_syn', 'tau2')
    )
    model = "exp2syn"
