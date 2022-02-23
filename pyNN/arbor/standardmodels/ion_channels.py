# ~/arbor/standardmodels/ion_channels.py

from pyNN.standardmodels import ion_channels as standard, build_translations

class NaChannel(standard.NaChannel):
    """
    Exactly the same as neuron.NaChannel because
    https://arbor.readthedocs.io/en/pydoc/mechanisms.html?highlight=huxley#density-mechanisms
    https://arbor.readthedocs.io/en/pydoc/nmodl.html#nmodl
    """
    translations = build_translations(("conductance_density", "gnabar"),  # (pynn_name, sim_name)
                                      ("e_rev", "ena"), )
    # variable_translations = {"m": ("hh", "m"),
    #                         "h": ("hh", "h"),}
    model = "hh"
    conductance_density_parameter = "gnabar"


class KdrChannel(standard.KdrChannel):
    translations = build_translations(('conductance_density', 'gkbar'),
                                      ('e_rev', 'ek'), )
    # variable_translations = {'n': ('hh', 'n')}
    model = "hh"
    conductance_density_parameter = 'gkbar'


class PassiveLeak(standard.PassiveLeak):
    translations = build_translations(('conductance_density', 'g'),
                                      ('e_rev', 'e'), )
    model = "pas"
    conductance_density_parameter = 'g'


class LeakyChannel(object):
    translations = build_translations(('conductance_density', 'gl'),
                                      ('e_rev', 'el'), )
    model = "hh"
    conductance_density_parameter = 'gl'

class CondExpPostSynapticResponse(standard.CondExpPostSynapticResponse):
    """
    Synapse with discontinuous change in conductance at an event followed by an exponential decay with time constant tau.
    """
    translations = build_translations(
        ('density', 'density'),
        ('e_rev', 'e'),
        ('tau_syn', 'tau')
    )
    model = "expsyn"
