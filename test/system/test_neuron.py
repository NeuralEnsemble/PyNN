from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.neuron
    have_neuron = True
except ImportError:
    have_neuron = False

def test_all():
    for scenario in (scenario1, scenario2):
        if have_neuron:
            yield scenario, pyNN.neuron
        else:
            raise SkipTest
