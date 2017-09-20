"""


"""

from pyNN.standardmodels import ion_channels as standard, build_translations


class NaChannel(standard.NaChannel):
    translations = build_translations(
        ('conductance_density', 'gnabar_hh'),
        ('e_rev', 'ena'),
    )
    variable_translations = {
        'm': ('hh', 'm'),
        'h': ('hh', 'h')
    }
    model = "hh"
    conductance_density_parameter = 'gnabar_hh'


class KdrChannel(standard.KdrChannel):
    translations = build_translations(
        ('conductance_density', 'gkbar_hh'),
        ('e_rev', 'ek'),
    )
    variable_translations = {
        'n': ('hh', 'n')
    }
    model = "hh"
    conductance_density_parameter = 'gkbar_hh'


class PassiveLeak(standard.PassiveLeak):
    translations = build_translations(
        ('conductance_density', 'g_pas'),
        ('e_rev', 'e_pas'),
    )
    model = "pas"
    conductance_density_parameter = 'g_pas'
