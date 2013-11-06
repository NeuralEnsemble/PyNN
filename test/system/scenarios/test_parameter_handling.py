
import numpy
from nose.tools import assert_equal
from pyNN.utility import assert_arrays_equal
from registry import register


@register()
def issue241(sim):
    spike_train1 = sim.Population(1, sim.SpikeSourcePoisson, {'rate' : [5], 'start' : [1000], 'duration': [1234]})
    spike_train2 = sim.Population(2, sim.SpikeSourcePoisson, {'rate' : [5, 6], 'start' : [1000, 1001], 'duration': [1234, 2345]})
    spike_train3 = sim.Population(1, sim.SpikeSourcePoisson, {'rate' : [5], 'start' : [1000], 'duration': 1234})
    spike_train4 = sim.Population(1, sim.SpikeSourcePoisson, {'rate' : [5], 'start' : [1000]})
    spike_train5 = sim.Population(2, sim.SpikeSourcePoisson, {'rate' : [5, 6], 'start' : [1000, 1001]})
    assert_arrays_equal(spike_train2.get('duration'), numpy.array([1234, 2345]))
    assert_equal(spike_train3.get(['rate', 'start', 'duration']), [5, 1000, 1234])


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    issue241(sim)
