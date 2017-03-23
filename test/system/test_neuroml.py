from nose.plugins.skip import SkipTest
from nose.tools import assert_equal
from .scenarios.registry import registry
import numpy

try:
    import pyNN.neuroml
    have_neuroml = True
except ImportError:
    have_neuroml = False

'''
###  Need to go through each of these and check which tests are appropriate
def test_scenarios():
    for scenario in registry:
        if "neuroml" not in scenario.exclude:
            scenario.description = "{}(neuroml)".format(scenario.__name__)
            if have_neuroml:
                yield scenario, pyNN.neuroml
            else:
                raise SkipTest'''


def test_save_validate_network():
    if not have_neuroml:
        raise SkipTest
    sim = pyNN.neuroml
    reference='Test0'

    sim.setup(reference=reference)
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=numpy.arange(10, 100, 10)))
    neurons = sim.Population(5, sim.IF_cond_exp(e_rev_I=-75))
    sim.end()
    
    from neuroml.utils import validate_neuroml2

    validate_neuroml2('%s.net.nml'%reference)
    
