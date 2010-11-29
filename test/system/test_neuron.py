from nose.plugins.skip import SkipTest
from scenarios import *

try:
    import pyNN.neuron
    have_neuron = True
except ImportError:
    have_neuron = False

def test_all():
    if have_neuron:
        scenario1(pyNN.neuron)
    else:
        raise SkipTest