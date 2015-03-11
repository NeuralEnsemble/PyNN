
import numpy
from numpy import nan
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal
from pyNN.utility import assert_arrays_equal, assert_arrays_almost_equal
from .registry import register


@register()
def issue241(sim):
    # "Nest SpikeSourcePoisson populations require all parameters to be passed to constructor"
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
    # "Setting attributes fails for Projections where either the pre- or post-synaptic Population has size 1"
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


@register()
def test_set_synaptic_parameters_fully_connected(sim):
    sim.setup()
    mpi_rank = sim.rank()
    p1 = sim.Population(4, sim.IF_cond_exp())
    p2 = sim.Population(2, sim.IF_cond_exp())
    syn = sim.TsodyksMarkramSynapse(U=0.5, weight=0.123, delay=0.1)
    prj = sim.Projection(p1, p2, sim.AllToAllConnector(), syn)

    expected = numpy.array([
        (0.0, 0.0, 0.123, 0.1, 0.5),
        (0.0, 1.0, 0.123, 0.1, 0.5),
        (1.0, 0.0, 0.123, 0.1, 0.5),
        (1.0, 1.0, 0.123, 0.1, 0.5),
        (2.0, 0.0, 0.123, 0.1, 0.5),
        (2.0, 1.0, 0.123, 0.1, 0.5),
        (3.0, 0.0, 0.123, 0.1, 0.5),
        (3.0, 1.0, 0.123, 0.1, 0.5),
    ])
    actual = numpy.array(prj.get(['weight', 'delay', 'U'], format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_arrays_almost_equal(actual[ind], expected, 1e-16)

    positional_weights = numpy.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=float)
    prj.set(weight=positional_weights)
    expected = positional_weights
    actual = prj.get('weight', format='array')
    if mpi_rank == 0:
        assert_arrays_equal(actual, expected)

    u_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    prj.set(U=u_list)
    expected = numpy.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]])
    actual = prj.get('U', format='array')
    if mpi_rank == 0:
        assert_arrays_equal(actual, expected)

    f_delay = lambda d: 0.5+d
    prj.set(delay=f_delay)
    expected = numpy.array([[0.5, 1.5], [1.5, 0.5], [2.5, 1.5], [3.5, 2.5]])
    actual = prj.get('delay', format='array')
    if mpi_rank == 0:
        assert_arrays_equal(actual, expected)

    # final sanity check
    expected = numpy.array([
        (0.0, 0.0, 0.0, 0.5, 0.9),
        (0.0, 1.0, 1.0, 1.5, 0.8),
        (1.0, 0.0, 2.0, 1.5, 0.7),
        (1.0, 1.0, 3.0, 0.5, 0.6),
        (2.0, 0.0, 4.0, 2.5, 0.5),
        (2.0, 1.0, 5.0, 1.5, 0.4),
        (3.0, 0.0, 6.0, 3.5, 0.3),
        (3.0, 1.0, 7.0, 2.5, 0.2),
    ])
    actual = numpy.array(prj.get(['weight', 'delay', 'U'], format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_arrays_equal(actual[ind], expected)
test_set_synaptic_parameters_fully_connected.__test__ = False



@register()
def test_set_synaptic_parameters_partially_connected(sim):
    sim.setup()
    mpi_rank = sim.rank()
    p1 = sim.Population(4, sim.IF_cond_exp())
    p2 = sim.Population(2, sim.IF_cond_exp())
    syn = sim.TsodyksMarkramSynapse(U=0.5, weight=0.123, delay=0.1)
    prj = sim.Projection(p1, p2, sim.FromListConnector([(0, 0) ,(3, 0), (1, 1), (1, 0), (2, 1)]), syn)

    expected = numpy.array([
        (0.0, 0.0, 0.123, 0.1, 0.5),
        (1.0, 0.0, 0.123, 0.1, 0.5),
        (1.0, 1.0, 0.123, 0.1, 0.5),
        (2.0, 1.0, 0.123, 0.1, 0.5),
        (3.0, 0.0, 0.123, 0.1, 0.5),
    ])
    actual = numpy.array(prj.get(['weight', 'delay', 'U'], format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_arrays_almost_equal(actual[ind], expected, 1e-16)

    positional_weights = numpy.array([[0, nan], [2, 3], [nan, 5], [6, nan]], dtype=float)
    prj.set(weight=positional_weights)
    expected = positional_weights
    actual = prj.get('weight', format='array')
    if mpi_rank == 0:
        assert_array_equal(actual, expected)

    u_list = [0.9, 0.8, 0.7, 0.6, 0.5]
    prj.set(U=u_list)
    expected = numpy.array([[0.9, nan], [0.8, 0.7], [nan, 0.6], [0.5, nan]])
    actual = prj.get('U', format='array')
    if mpi_rank == 0:
        assert_array_equal(actual, expected)

    f_delay = lambda d: 0.5+d
    prj.set(delay=f_delay)
    expected = numpy.array([[0.5, nan], [1.5, 0.5], [nan, 1.5], [3.5, nan]])
    actual = prj.get('delay', format='array')
    if mpi_rank == 0:
        assert_array_equal(actual, expected)

    # final sanity check
    expected = numpy.array([
        (0.0, 0.0, 0.0, 0.5, 0.9),
        (1.0, 0.0, 2.0, 1.5, 0.8),
        (1.0, 1.0, 3.0, 0.5, 0.7),
        (2.0, 1.0, 5.0, 1.5, 0.6),
        (3.0, 0.0, 6.0, 3.5, 0.5),
    ])
    actual = numpy.array(prj.get(['weight', 'delay', 'U'], format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_array_equal(actual[ind], expected)
test_set_synaptic_parameters_partially_connected.__test__ = False



@register()
def test_set_synaptic_parameters_multiply_connected(sim):
    sim.setup()
    mpi_rank = sim.rank()
    p1 = sim.Population(4, sim.IF_cond_exp())
    p2 = sim.Population(2, sim.IF_cond_exp())
    syn = sim.TsodyksMarkramSynapse(U=0.5, weight=0.123, delay=0.1)
    prj = sim.Projection(p1, p2, sim.FromListConnector([(0, 0), (1, 0), (3, 0), (1, 1), (1, 0), (2, 1)]), syn)

    expected = numpy.array([
        (0.0, 0.0, 0.123, 0.1, 0.5),
        (1.0, 0.0, 0.123, 0.1, 0.5),
        (1.0, 0.0, 0.123, 0.1, 0.5),
        (1.0, 1.0, 0.123, 0.1, 0.5),
        (2.0, 1.0, 0.123, 0.1, 0.5),
        (3.0, 0.0, 0.123, 0.1, 0.5),
    ])
    actual = numpy.array(prj.get(['weight', 'delay', 'U'], format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_arrays_almost_equal(actual[ind], expected, 1e-16)

    positional_weights = numpy.array([[0, nan], [2, 3], [nan, 5], [6, nan]], dtype=float)
    prj.set(weight=positional_weights)
    expected = numpy.array([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0),
        (1.0, 0.0, 2.0),
        (1.0, 1.0, 3.0),
        (2.0, 1.0, 5.0),
        (3.0, 0.0, 6.0),
    ])
    actual = numpy.array(prj.get('weight', format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_arrays_almost_equal(actual[ind], expected, 1e-16)

    # postponing implementation of this functionality until after 0.8.0
    # u_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    # prj.set(U=u_list)
    # expected = numpy.array([
    #     (0.0, 0.0, 0.9),
    #     (1.0, 0.0, 0.8),
    #     (1.0, 0.0, 0.7),
    #     (1.0, 1.0, 0.6),
    #     (2.0, 1.0, 0.5),
    #     (3.0, 0.0, 0.4),
    # ])
    # actual = numpy.array(prj.get('U', format='list'))
    # if mpi_rank == 0:
    #     ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
    #     assert_arrays_almost_equal(actual[ind], expected, 1e-16)

    f_delay = lambda d: 0.5+d
    prj.set(delay=f_delay)
    expected = numpy.array([
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 1.5),
        (1.0, 0.0, 1.5),
        (1.0, 1.0, 0.5),
        (2.0, 1.0, 1.5),
        (3.0, 0.0, 3.5),
    ])
    actual = numpy.array(prj.get('delay', format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_arrays_almost_equal(actual[ind], expected, 1e-16)

    # final sanity check
    expected = numpy.array([
        (0.0, 0.0, 0.0, 0.5, 0.5),
        (1.0, 0.0, 2.0, 1.5, 0.5),
        (1.0, 0.0, 2.0, 1.5, 0.5),
        (1.0, 1.0, 3.0, 0.5, 0.5),
        (2.0, 1.0, 5.0, 1.5, 0.5),
        (3.0, 0.0, 6.0, 3.5, 0.5),
    ])
    actual = numpy.array(prj.get(['weight', 'delay', 'U'], format='list'))
    if mpi_rank == 0:
        ind = numpy.lexsort((actual[:, 1], actual[:, 0]))
        assert_array_equal(actual[ind], expected)
test_set_synaptic_parameters_multiply_connected.__test__ = False



if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    issue241(sim)
    issue302(sim)
    test_set_synaptic_parameters_fully_connected(sim)
    test_set_synaptic_parameters_partially_connected(sim)
