
import numpy
from nose.tools import assert_equal
from pyNN.utility import assert_arrays_equal
from registry import register


@register()
def issue241(sim):
    sim.setup()
    spike_train1 = sim.Population(1, sim.SpikeSourcePoisson, {'rate' : [5], 'start' : [1000], 'duration': [1234]})
    spike_train2 = sim.Population(2, sim.SpikeSourcePoisson, {'rate' : [5, 6], 'start' : [1000, 1001], 'duration': [1234, 2345]})
    spike_train3 = sim.Population(1, sim.SpikeSourcePoisson, {'rate' : [5], 'start' : [1000], 'duration': 1234})
    spike_train4 = sim.Population(1, sim.SpikeSourcePoisson, {'rate' : [5], 'start' : [1000]})
    spike_train5 = sim.Population(2, sim.SpikeSourcePoisson, {'rate' : [5, 6], 'start' : [1000, 1001]})
    assert_arrays_equal(spike_train2.get('duration'), numpy.array([1234, 2345]))
    assert_equal(spike_train3.get(['rate', 'start', 'duration']), [5, 1000, 1234])
    sim.end()


@register()
def issue302(sim):
    sim.setup()
    p1 = sim.Population(1, sim.IF_cond_exp())
    p5 = sim.Population(5, sim.IF_cond_exp())
    prj15 = sim.Projection(p1, p5, sim.AllToAllConnector())
    prj51 = sim.Projection(p5, p1, sim.AllToAllConnector())
    prj55 = sim.Projection(p5, p5, sim.AllToAllConnector())
    prj15.set(weight=0.123)
    prj51.set(weight=0.123)
    prj55.set(weight=0.123)
    sim.end()

if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    issue241(sim)
