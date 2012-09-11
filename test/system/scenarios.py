# encoding: utf-8
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN import common, recording
from pyNN.space import Space, Grid3D, RandomStructure, Cuboid
from nose.tools import assert_equal
import glob, os
import numpy
import quantities as pq
from pyNN.utility import init_logging, assert_arrays_equal, assert_arrays_almost_equal

import logging
logger = logging.getLogger("TEST")

scenarios = []
def register(exclude=[]):
    def inner_register(scenario):
        #print "registering %s with exclude=%s" % (scenario, exclude)
        if scenario not in scenarios:
            scenario.exclude = exclude
            scenarios.append(scenario)
        return scenario
    return inner_register


@register(exclude=["nemo"])
def scenario1(sim):
    """
    Balanced network of integrate-and-fire neurons.
    """
    cell_params = {
        'tau_m': 20.0, 'tau_syn_E': 5.0, 'tau_syn_I': 10.0, 'v_rest': -60.0,
        'v_reset': -60.0, 'v_thresh': -50.0, 'cm': 1.0, 'tau_refrac': 5.0,
        'e_rev_E': 0.0, 'e_rev_I': -80.0
    }
    stimulation_params = {'rate' : 100.0, 'duration' : 50.0}
    n_exc = 80
    n_inh = 20
    n_input = 20
    rngseed = 98765
    parallel_safe = True
    n_threads = 1
    pconn_recurr = 0.02
    pconn_input = 0.01
    tstop = 1000.0
    delay = 0.2
    dt    = 0.1
    weights = {
        'excitatory': 4.0e-3,
        'inhibitory': 51.0e-3,
        'input': 0.1,
    }

    sim.setup(timestep=dt, min_delay=dt, threads=n_threads)
    all_cells = sim.Population(n_exc+n_inh, sim.IF_cond_exp(**cell_params), label="All cells")
    cells = {
        'excitatory': all_cells[:n_exc],
        'inhibitory': all_cells[n_exc:],
        'input': sim.Population(n_input, sim.SpikeSourcePoisson(**stimulation_params), label="Input")
    }

    rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
    uniform_distr = RandomDistribution(
                        'uniform',
                        [cell_params['v_reset'], cell_params['v_thresh']],
                        rng=rng)
    all_cells.initialize(v=uniform_distr)

    connections = {}
    for name, pconn, target in (
        ('excitatory', pconn_recurr, 'excitatory'),
        ('inhibitory', pconn_recurr, 'inhibitory'),
        ('input',      pconn_input,  'excitatory'),
    ):
        connector = sim.FixedProbabilityConnector(pconn, weights=weights[name], delays=delay)
        connections[name] = sim.Projection(cells[name], all_cells, connector,
                                           target=target, label=name, rng=rng)

    all_cells.record('spikes')
    cells['excitatory'][0:2].record('v')
    assert_equal(cells['excitatory'][0:2].grandparent, all_cells)

    sim.run(tstop)

    E_count = cells['excitatory'].mean_spike_count()
    I_count = cells['inhibitory'].mean_spike_count()
    print "Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,)
    print "Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,)
    sim.end()


@register(exclude=["nemo"])
def scenario1a(sim):
    """
    Balanced network of integrate-and-fire neurons, built with the "low-level"
    API.
    """
    cell_params = {
        'tau_m': 10.0, 'tau_syn_E': 2.0, 'tau_syn_I': 5.0, 'v_rest': -60.0,
        'v_reset': -65.0, 'v_thresh': -55.0, 'cm': 0.5, 'tau_refrac': 2.5,
        'e_rev_E': 0.0, 'e_rev_I': -75.0
    }
    stimulation_params = {'rate': 80.0, 'duration': 50.0}
    n_exc = 80
    n_inh = 20
    n_input = 20
    rngseed = 87546
    parallel_safe = True
    n_threads = 1
    pconn_recurr = 0.03
    pconn_input = 0.01
    tstop = 1000.0
    delay = 1
    w_exc = 3.0e-3
    w_inh = 45.0e-3
    w_input = 0.12
    dt      = 0.1

    sim.setup(timestep=dt, min_delay=dt, threads=n_threads)
    iaf_neuron = sim.IF_cond_alpha(**cell_params)
    excitatory_cells = sim.create(iaf_neuron, n=n_exc)
    inhibitory_cells = sim.create(iaf_neuron, n=n_inh)
    inputs = sim.create(sim.SpikeSourcePoisson(**stimulation_params), n=n_input)
    all_cells = excitatory_cells + inhibitory_cells
    sim.initialize(all_cells, v=cell_params['v_rest'])

    sim.connect(excitatory_cells, all_cells, weight=w_exc, delay=delay,
                synapse_type='excitatory', p=pconn_recurr)
    sim.connect(inhibitory_cells, all_cells, weight=w_exc, delay=delay,
                synapse_type='inhibitory', p=pconn_recurr)
    sim.connect(inputs, all_cells, weight=w_input, delay=delay,
                synapse_type='excitatory', p=pconn_input)
    sim.record('spikes', all_cells, "scenario1a_%s_spikes.pkl" % sim.__name__)
    sim.record('v', excitatory_cells[0:2], "scenario1a_%s_v.pkl" % sim.__name__)

    sim.run(tstop)

    E_count = excitatory_cells.mean_spike_count()
    I_count = inhibitory_cells.mean_spike_count()
    print "Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,)
    print "Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,)
    sim.end()
    for filename in glob.glob("scenario1a_*"):
        os.remove(filename)


@register(exclude=["moose", "nemo"])
def scenario2(sim):
    """
    Array of neurons, each injected with a different current.

    firing period of a IF neuron injected with a current I:

    T = tau_m*log(I*tau_m/(I*tau_m - v_thresh*cm))

    (if v_rest = v_reset = 0.0)

    we set the refractory period to be very large, so each neuron fires only
    once (except neuron[0], which never reaches threshold).
    """
    n = 100
    t_start = 25.0
    duration = 100.0
    t_stop = 150.0
    tau_m = 20.0
    v_thresh = 10.0
    cm = 1.0
    cell_params = {"tau_m": tau_m, "v_rest": 0.0, "v_reset": 0.0,
                   "tau_refrac": 100.0, "v_thresh": v_thresh, "cm": cm}
    I0 = (v_thresh*cm)/tau_m
    sim.setup(timestep=0.01, spike_precision="off_grid")
    neurons = sim.Population(n, sim.IF_curr_exp(**cell_params))
    neurons.initialize(v=0.0)
    I = numpy.arange(I0, I0+1.0, 1.0/n)
    currents = [sim.DCSource(start=t_start, stop=t_start+duration, amplitude=amp)
                for amp in I]
    for j, (neuron, current) in enumerate(zip(neurons, currents)):
        if j%2 == 0:                      # these should
            neuron.inject(current)        # be entirely
        else:                             # equivalent
            current.inject_into([neuron])
    neurons.record(['spikes', 'v'])

    sim.run(t_stop)

    spiketrains = neurons.get_data().segments[0].spiketrains
    assert_equal(len(spiketrains), n)
    assert_equal(len(spiketrains[0]), 0) # first cell does not fire
    assert_equal(len(spiketrains[1]), 1) # other cells fire once
    assert_equal(len(spiketrains[-1]), 1) # other cells fire once
    expected_spike_times = t_start + tau_m*numpy.log(I*tau_m/(I*tau_m - v_thresh*cm))
    a = spike_times = [numpy.array(st)[0] for st in spiketrains[1:]]
    b = expected_spike_times[1:]
    max_error = abs((a-b)/b).max()
    print "max error =", max_error
    assert max_error < 0.005, max_error
    sim.end()
    return a,b, spike_times


@register(exclude=["moose", "nemo", "brian"])
def scenario3(sim):
    """
    Simple feed-forward network network with additive STDP. The second half of
    the presynaptic neurons fires faster than the second half, so their
    connections should be potentiated more.
    """

    init_logging(logfile=None, debug=True)
    second = 1000.0
    duration = 10
    tau_m = 20 # ms
    cm = 1.0 # nF
    v_reset = -60
    cell_parameters = dict(
        tau_m = tau_m,
        cm = cm,
        v_rest = -70,
        e_rev_E = 0,
        e_rev_I = -70,
        v_thresh = -54,
        v_reset = v_reset,
        tau_syn_E = 5,
        tau_syn_I = 5,
    )
    g_leak = cm/tau_m # µS

    w_min = 0.0*g_leak
    w_max = 0.05*g_leak

    r1 = 5.0
    r2 = 40.0

    sim.setup()
    pre = sim.Population(100, sim.SpikeSourcePoisson())
    post = sim.Population(10, sim.IF_cond_exp())

    pre.set(duration=duration*second)
    pre.set(start=0.0)
    pre[:50].set(rate=r1)
    pre[50:].set(rate=r2)
    assert_equal(pre[49].rate, r1)
    assert_equal(pre[50].rate, r2)
    post.set(**cell_parameters)
    post.initialize(v=RandomDistribution('normal', (v_reset, 5.0)))

    stdp = sim.SynapseDynamics(
                slow=sim.STDPMechanism(
                        sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0 ),
                        sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max,
                                                     A_plus=0.01, A_minus=0.01),
                        #dendritic_delay_fraction=0.5))
                        dendritic_delay_fraction=1))

    connections = sim.Projection(pre, post, sim.AllToAllConnector(),
                                 target='excitatory', synapse_dynamics=stdp)

    initial_weight_distr = RandomDistribution('uniform', (w_min, w_max))
    connections.randomizeWeights(initial_weight_distr)
    initial_weights = connections.get('weights', format='array')
    assert initial_weights.min() >= w_min
    assert initial_weights.max() < w_max
    assert initial_weights[0,0] != initial_weights[1,0]

    pre.record('spikes')
    post.record('spikes')
    post[0:1].record('v')

    sim.run(duration*second)

    actual_rate = pre.mean_spike_count()/duration
    expected_rate = (r1+r2)/2
    errmsg = "actual rate: %g  expected rate: %g" % (actual_rate, expected_rate)
    assert abs(actual_rate - expected_rate) < 1, errmsg
    #assert abs(pre[:50].mean_spike_count()/duration - r1) < 1
    #assert abs(pre[50:].mean_spike_count()/duration- r2) < 1
    final_weights = connections.get('weights', format='array')
    assert initial_weights[0,0] != final_weights[0,0]

    import scipy.stats
    t,p = scipy.stats.ttest_ind(initial_weights[:50,:].flat, initial_weights[50:,:].flat)
    assert p > 0.05, p
    t,p = scipy.stats.ttest_ind(final_weights[:50,:].flat, final_weights[50:,:].flat)
    assert p < 0.01, p
    assert final_weights[:50,:].mean() < final_weights[50:,:].mean()

    return initial_weights, final_weights, pre, post, connections


@register(exclude=["nemo"])
def ticket166(sim, interactive=False):
    """
    Check that changing the spike_times of a SpikeSourceArray mid-simulation
    works (see http://neuralensemble.org/trac/PyNN/ticket/166)
    """
    dt = 0.1 # ms
    t_step = 100.0 # ms
    lag = 3.0 # ms

    if interactive:
        import matplotlib.pyplot as plt
        plt.ion()

    sim.setup(timestep=dt, min_delay=dt)

    spikesources = sim.Population(2, sim.SpikeSourceArray())
    cells = sim.Population(2, sim.IF_cond_exp())
    conn = sim.Projection(spikesources, cells, sim.OneToOneConnector(weights=0.01))
    cells.record('v')

    spiketimes = numpy.arange(2.0, t_step, t_step/13.0)
    spikesources[0].spike_times = spiketimes
    spikesources[1].spike_times = spiketimes + lag

    t = sim.run(t_step) # both neurons depolarized by synaptic input
    t = sim.run(t_step) # no more synaptic input, neurons decay

    spiketimes += 2*t_step
    spikesources[0].spike_times = spiketimes
    # note we add no new spikes to the second source
    t = sim.run(t_step) # first neuron gets depolarized again

    vm = cells.get_data().segments[0].analogsignalarrays[0]
    final_v_0 = vm[-1, 0]
    final_v_1 = vm[-1, 1]

    sim.end()

    if interactive:
        plt.plot(vm.times, vm[:, 0])
        plt.plot(vm.times, vm[:, 1])
        plt.savefig("ticket166_%s.png" % sim.__name__)

    assert final_v_0 > -60.0  # first neuron has been depolarized again
    assert final_v_1 < -64.99 # second neuron has decayed back towards rest


@register()
def test_reset(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    dt      = 1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
    p.record('v')

    for i in range(repeats):
        sim.run(10.0)
        sim.reset()
    data = p.get_data(clear=False)
    sim.end()

    assert len(data.segments) == repeats
    for segment in data.segments[1:]:
        assert_arrays_almost_equal(segment.analogsignalarrays[0],
                                   data.segments[0].analogsignalarrays[0], 1e-11)


@register()
def test_reset_with_clear(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    dt      = 1
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
    p.record('v')

    data = []
    for i in range(repeats):
        sim.run(10.0)
        data.append(p.get_data(clear=True))
        sim.reset()

    sim.end()

    for rec in data:
        assert len(rec.segments) == 1
        assert_arrays_almost_equal(rec.segments[0].analogsignalarrays[0],
                                   data[0].segments[0].analogsignalarrays[0], 1e-11)


@register(exclude=['brian', 'pcsim', 'nemo'])
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
    ti = lambda i: data.segments[i].analogsignalarrays[0].times
    assert_arrays_equal(ti(0), ti(1))
    idx = lambda i: data.segments[i].analogsignalarrays[0].channel_indexes
    assert idx(0) == [3]
    assert idx(1) == [4]
    vi = lambda i: data.segments[i].analogsignalarrays[0]
    assert vi(0).shape == vi(1).shape == (101, 1)
    assert vi(0)[0, 0] == vi(1)[0, 0] == p.initial_values['v'].evaluate(simplify=True)*pq.mV # the first value should be the same
    assert not (vi(0)[1:, 0] == vi(1)[1:, 0]).any()            # none of the others should be, because of different i_offset


@register()
def test_setup(sim):
    """
    Run the same simulation n times, recreating the network each time,
    and check the results are the same each time.
    """
    n = 3
    data = []
    dt   = 1

    for i in range(n):
        sim.setup(timestep=dt, min_delay=dt)
        p = sim.Population(1, sim.IF_curr_exp(i_offset=0.1))
        p.record('v')
        sim.run(10.0)
        data.append(p.get_data())
        sim.end()

    assert len(data) == n
    for block in data:
        assert len(block.segments) == 1
        signals = block.segments[0].analogsignalarrays
        assert len(signals) == 1
        assert_arrays_equal(signals[0], data[0].segments[0].analogsignalarrays[0])


@register(exclude=['pcsim', 'moose', 'nemo'])
def test_EIF_cond_alpha_isfa_ista(sim):
    sim.setup(timestep=0.01, min_delay=0.1, max_delay=4.0)
    ifcell = sim.create(sim.EIF_cond_alpha_isfa_ista(
                            i_offset=1.0, tau_refrac=2.0, v_spike=-40))
    ifcell.record(['spikes', 'v'])
    ifcell.initialize(v=-65)
    sim.run(200.0)
    data = ifcell.get_data().segments[0]
    expected_spike_times = numpy.array([10.02, 25.52, 43.18, 63.42, 86.67,  113.13, 142.69, 174.79]) * pq.ms
    diff = (data.spiketrains[0] - expected_spike_times)/expected_spike_times
    assert abs(diff).max() < 0.001
    sim.end()
    return data


@register(exclude=['pcsim', 'nemo', 'brian'])
def test_HH_cond_exp(sim):
    sim.setup(timestep=0.001, min_delay=0.1)
    cellparams = {
        'gbar_Na'   : 20.0,
        'gbar_K'    : 6.0,
        'g_leak'    : 0.01,
        'cm'        : 0.2,
        'v_offset'  : -63.0,
        'e_rev_Na'  : 50.0,
        'e_rev_K'   : -90.0,
        'e_rev_leak': -65.0,
        'e_rev_E'   : 0.0,
        'e_rev_I'   : -80.0,
        'tau_syn_E' : 0.2,
        'tau_syn_I' : 2.0,
        'i_offset'  : 1.0,
    }
    hhcell = sim.create(sim.HH_cond_exp(**cellparams))
    sim.initialize(hhcell, v=-64.0)
    hhcell.record('v')
    sim.run(20.0)
    v = hhcell.get_data().segments[0].filter(name='v')[0]
    first_spike = v.times[numpy.where(v>0)[0][0]]
    assert first_spike/pq.ms - 2.95 < 0.01


@register(exclude=['pcsim', 'moose', 'nemo'])
def test_record_vm_and_gsyn_from_assembly(sim):
    from pyNN.utility import init_logging
    init_logging(logfile=None, debug=True)
    dt    = 0.1
    tstop = 100.0
    sim.setup(timestep=dt, min_delay=dt)
    cells = sim.Population(5, sim.IF_cond_exp()) + sim.Population(6, sim.EIF_cond_exp_isfa_ista())
    inputs = sim.Population(5, sim.SpikeSourcePoisson(rate=50.0))
    sim.connect(inputs, cells, weight=0.1, delay=0.5, synapse_type='inhibitory')
    sim.connect(inputs, cells, weight=0.1, delay=0.3, synapse_type='excitatory')
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

    assert_equal(vm_p0.channel_indexes, range(5))
    assert_equal(vm_p1.channel_indexes, range(6))
    assert_equal(vm_all.channel_indexes, range(11))
    assert_equal(gsyn_p0.channel_indexes, [ 2, 3, 4])
    assert_equal(gsyn_p1.channel_indexes, [ 0, 1, 2, 3])
    assert_equal(gsyn_all.channel_indexes, range(2,9))

    sim.end()


@register(exclude=["pcsim", "nemo"])
def test_changing_electrode(sim):
    """
    Check that changing the values of the electrodes on the fly is taken into account
    """
    repeats = 2
    dt      = 0.1
    simtime = 100
    sim.setup(timestep=dt, min_delay=dt)
    p = sim.Population(1, sim.IF_curr_exp())
    c = sim.DCSource(amplitude=0.0)
    c.inject_into(p)
    p.record('v')

    for i in range(repeats):
        sim.run(simtime)
        c.amplitude += 0.1

    data = p.get_data().segments[0].analogsignalarrays[0]

    sim.end()

    # check that the value of v just before increasing the current is less than
    # the value at the end of the simulation
    assert data[int(simtime/dt), 0] < data[-1, 0]



@register(exclude=['nemo'])
def ticket195(sim):
    """
    Check that the `connect()` function works correctly with single IDs (see
    http://neuralensemble.org/trac/PyNN/ticket/195)
    """
    init_logging(None, debug=True)
    sim.setup(timestep=0.01)
    pre = sim.Population(10, sim.SpikeSourceArray(spike_times=range(1,10)))
    post = sim.Population(10, sim.IF_cond_exp())
    sim.connect(pre[0], post[0], weight=0.01, delay=0.1, p=1)
    #prj = sim.Projection(pre, post, sim.FromListConnector([(0, 0, 0.01, 0.1)]))
    post.record('spikes')
    sim.run(100.0)
    assert_arrays_almost_equal(post.get_data().segments[0].spiketrains[0], numpy.array([13.4])*pq.ms, 0.5)


@register()
def ticket226(sim):
    """
    Check that the start time of DCSources is correctly taken into account
    http://neuralensemble.org/trac/PyNN/ticket/226)
    """
    sim.setup(timestep=0.1)

    cell = sim.Population(1, sim.IF_curr_alpha(tau_m=20.0, cm=1.0, v_rest=-60.0,
                                               v_reset=-60.0))
    cell.initialize(v=-60.0)
    inj = sim.DCSource(amplitude=1.0, start=10.0, stop=20.0)
    cell.inject(inj)
    cell.record_v()
    sim.run(30.0)
    v = cell.get_data().segments[0].filter(name='v')[0][:, 0]
    v_10p0 = v[abs(v.times-10.0*pq.ms)<0.01*pq.ms][0]
    assert abs(v_10p0 - -60.0*pq.mV) < 1e-10
    v_10p1 = v[abs(v.times-10.1*pq.ms)<0.01*pq.ms][0]
    assert v_10p1 > -59.99*pq.mV, v_10p1


@register(exclude=["nemo", "brian"])
def scenario4(sim):
    """
    Network with spatial structure
    """
    init_logging(logfile=None, debug=True)
    sim.setup()
    rng = NumpyRNG(seed=76454, parallel_safe=False)

    input_layout = RandomStructure(boundary=Cuboid(width=500.0, height=500.0, depth=100.0),
                                   origin=(0, 0, 0), rng=rng)
    inputs = sim.Population(100, sim.SpikeSourcePoisson(rate=RandomDistribution('uniform', [3.0, 7.0], rng=rng)),
                            structure=input_layout, label="inputs")
    output_layout = Grid3D(aspect_ratioXY=1.0, aspect_ratioXZ=5.0, dx=10.0, dy=10.0, dz=10.0,
                           x0=0.0, y0=0.0, z0=200.0)
    outputs = sim.Population(200, sim.EIF_cond_exp_isfa_ista(),
                             initial_values = {'v': RandomDistribution('normal', [-65.0, 5.0], rng=rng),
                                               'w': RandomDistribution('normal', [0.0, 1.0], rng=rng)},
                             structure=output_layout, # 10x10x2 grid
                             label="outputs")
    logger.debug("Output population positions:\n %s", outputs.positions)
    DDPC = sim.DistanceDependentProbabilityConnector
    input_connectivity = DDPC("0.5*exp(-d/100.0)",
                             weights=RandomDistribution('normal', (0.1, 0.02), rng=rng),
                             delays="0.5 + d/100.0",
                             space=Space(axes='xy'))
    recurrent_connectivity = DDPC("sin(pi*d/250.0)**2",
                                  weights=0.05,
                                  delays="0.2 + d/100.0",
                                  space=Space(periodic_boundaries=((-100.0, 100.0), (-100.0, 100.0), None))) # should add "calculate_boundaries" method to Structure classes
    depressing = sim.SynapseDynamics(fast=sim.TsodyksMarkramMechanism(U=0.5, tau_rec=800.0, tau_facil=0.0))
    facilitating = sim.SynapseDynamics(fast=sim.TsodyksMarkramMechanism(U=0.04, tau_rec=100.0, tau_facil=1000.0))
    input_connections = sim.Projection(inputs, outputs, input_connectivity,
                                       target='excitatory',
                                       synapse_dynamics=depressing,
                                       label="input connections",
                                       rng=rng)
    recurrent_connections = sim.Projection(outputs, outputs, recurrent_connectivity,
                                           target='inhibitory',
                                           synapse_dynamics=facilitating,
                                           label="recurrent connections",
                                           rng=rng)
    outputs.record('spikes')
    outputs.sample(10, rng=rng).record('v')
    sim.run(1000.0)
    data = outputs.get_data()
    sim.end()
    return data
