
import numpy
import quantities as pq
from nose.tools import assert_equal
from pyNN.utility import assert_arrays_equal, assert_arrays_almost_equal, init_logging
from registry import register


@register(exclude=['pcsim', 'nemo'])
def test_reset_recording(sim):
    """
    Check that record(None) resets the list of things to record.

    This test injects different levels of current into two neurons. In the
    first run, we record one of the neurons, in the second we record the other.
    The main point is to check that the first neuron is not recorded in the
    second run.
    """
    sim.setup()
    p = sim.Population(7, sim.IF_cond_exp())
    p[3].i_offset = 0.1
    p[4].i_offset = 0.2
    p[3:4].record('v')
    sim.run(10.0)
    sim.reset()
    p.record(None)
    p[4:5].record('v')
    sim.run(10.0)
    data = p.get_data()
    sim.end()
    ti = lambda i: data.segments[i].analogsignalarrays[0].times
    assert_arrays_equal(ti(0), ti(1))
    idx = lambda i: data.segments[i].analogsignalarrays[0].channel_index
    assert idx(0) == [3]
    assert idx(1) == [4]
    vi = lambda i: data.segments[i].analogsignalarrays[0]
    assert vi(0).shape == vi(1).shape == (101, 1)
    assert vi(0)[0, 0] == vi(1)[0, 0] == p.initial_values['v'].evaluate(simplify=True)*pq.mV # the first value should be the same
    assert not (vi(0)[1:, 0] == vi(1)[1:, 0]).any()            # none of the others should be, because of different i_offset


@register(exclude=['pcsim', 'moose', 'nemo'])
def test_record_vm_and_gsyn_from_assembly(sim):
    from pyNN.utility import init_logging
    init_logging(logfile=None, debug=True)
    dt    = 0.1
    tstop = 100.0
    sim.setup(timestep=dt, min_delay=dt)
    cells = sim.Population(5, sim.IF_cond_exp()) + sim.Population(6, sim.EIF_cond_exp_isfa_ista())
    inputs = sim.Population(5, sim.SpikeSourcePoisson(rate=50.0))
    sim.connect(inputs, cells, weight=0.1, delay=0.5, receptor_type='inhibitory')
    sim.connect(inputs, cells, weight=0.1, delay=0.3, receptor_type='excitatory')
    cells.record('v')
    cells[2:9].record(['gsyn_exc', 'gsyn_inh'])
#    for p in cells.populations:
#        assert_equal(p.recorders['v'].recorded, set(p.all_cells))

#    assert_equal(cells.populations[0].recorders['gsyn'].recorded, set(cells.populations[0].all_cells[2:5]))
#    assert_equal(cells.populations[1].recorders['gsyn'].recorded, set(cells.populations[1].all_cells[0:4]))
    sim.run(tstop)
    data0 = cells.populations[0].get_data().segments[0]
    data1 = cells.populations[1].get_data().segments[0]
    data_all = cells.get_data().segments[0]
    vm_p0 = data0.filter(name='v')[0]
    vm_p1 = data1.filter(name='v')[0]
    vm_all = data_all.filter(name='v')[0]
    gsyn_p0 = data0.filter(name='gsyn_exc')[0]
    gsyn_p1 = data1.filter(name='gsyn_exc')[0]
    gsyn_all = data_all.filter(name='gsyn_exc')[0]

    n_points = int(tstop/dt) + 1
    assert_equal(vm_p0.shape, (n_points, 5))
    assert_equal(vm_p1.shape, (n_points, 6))
    assert_equal(vm_all.shape, (n_points, 11))
    assert_equal(gsyn_p0.shape, (n_points, 3))
    assert_equal(gsyn_p1.shape, (n_points, 4))
    assert_equal(gsyn_all.shape, (n_points, 7))

    assert_arrays_equal(vm_p1[:,3], vm_all[:,8])

    assert_arrays_equal(vm_p0.channel_index, numpy.arange(5))
    assert_arrays_equal(vm_p1.channel_index, numpy.arange(6))
    assert_arrays_equal(vm_all.channel_index, numpy.arange(11))
    assert_arrays_equal(gsyn_p0.channel_index, numpy.array([ 2, 3, 4]))
    assert_arrays_equal(gsyn_p1.channel_index, numpy.arange(4))
    assert_arrays_equal(gsyn_all.channel_index, numpy.arange(2, 9))

    sim.end()


@register()
def issue259(sim):
    """
    A test that retrieving data with "clear=True" gives correct spike trains.
    """
    sim.setup(time_step=0.05, spike_precision="off_grid")
    p = sim.Population(1, sim.SpikeSourceArray(spike_times=[0.025, 10.025, 12.34, 1000.025]))
    p.record('spikes')
    sim.run(10.0)
    spiketrains0 = p.get_data('spikes', clear=True).segments[0].spiketrains
    print spiketrains0[0]
    sim.run(10.0)
    spiketrains1 = p.get_data('spikes', clear=True).segments[0].spiketrains
    print spiketrains1[0]
    sim.run(10.0)
    spiketrains2 = p.get_data('spikes', clear=True).segments[0].spiketrains
    print spiketrains2[0]
    sim.end()

    assert_arrays_almost_equal(spiketrains0[0], numpy.array([0.025])*pq.ms, 1e-17)
    assert_arrays_almost_equal(spiketrains1[0], numpy.array([10.025, 12.34])*pq.ms, 1e-17)
    assert_equal(spiketrains2[0].size, 0)


if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    test_reset_recording(sim)
    test_record_vm_and_gsyn_from_assembly(sim)
    issue259(sim)
