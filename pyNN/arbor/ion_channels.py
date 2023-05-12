"""


"""

from pyNN.standardmodels import ion_channels as standard, build_translations


class NaChannel(standard.NaChannel):
    translations = build_translations(
        ('conductance_density', 'gnabar'),
    )
    variable_translations = {
        'm': ('hh', 'm'),
        'h': ('hh', 'h')
    }
    model = "hh"
    conductance_density_parameter = 'gnabar'


class KdrChannel(standard.KdrChannel):
    translations = build_translations(
        ('conductance_density', 'gkbar'),
    )
    variable_translations = {
        'n': ('hh', 'n')
    }
    model = "hh"
    conductance_density_parameter = 'gkbar'


class PassiveLeak(standard.PassiveLeak):
    translations = build_translations(
        ('conductance_density', 'g'),
        ('e_rev', 'e'),
    )
    model = "pas"
    conductance_density_parameter = 'g'
