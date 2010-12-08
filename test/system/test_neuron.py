from nose.plugins.skip import SkipTest
from scenarios import scenarios
from nose.tools import assert_equal, assert_almost_equal

try:
    import pyNN.neuron
    have_neuron = True
except ImportError:
    have_neuron = False

def test_scenarios():
    for scenario in scenarios:
        scenario.description = scenario.__name__
        if have_neuron:
            yield scenario, pyNN.neuron
        else:
            raise SkipTest

def test_ticket168():
    """
    Error setting firing rate of `SpikeSourcePoisson` after `reset()` in NEURON
    http://neuralensemble.org/trac/PyNN/ticket/168
    """
    pynn = pyNN.neuron
    pynn.setup()
    cell = pynn.Population(1, cellclass=pynn.SpikeSourcePoisson, label="cell")
    cell[0].rate = 12
    pynn.run(10.)
    pynn.reset()
    cell[0].rate = 12
    pynn.run(10.)
    assert_almost_equal(pynn.get_current_time(), 10.0, places=11)
    assert_equal(cell[0]._cell.interval, 1000.0/12.0)
    