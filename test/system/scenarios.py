from pyNN.random import NumpyRNG, RandomDistribution
from pyNN import common, recording
from nose.tools import assert_equal
import numpy
from pyNN.utility import assert_arrays_equal, assert_arrays_almost_equal, sort_by_column

def set_simulator(sim):
    common.simulator = sim.simulator
    recording.simulator = sim.simulator

scenarios = []

def register(scenario):
    if scenario not in scenarios:
        scenarios.append(scenario)
    return scenario

@register
def scenario1(sim):
    """
    Balanced network of integrate-and-fire neurons.
    """
    set_simulator(sim)
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
    dt = 0.1
    pconn_recurr = 0.02
    pconn_input = 0.01
    tstop = 1000.0
    delay = 0.2
    weights = {
        'excitatory': 4.0e-3,
        'inhibitory': 51.0e-3,
        'input': 0.1,
    }
       
    sim.setup(timestep=0.1, threads=n_threads)
    all_cells = sim.Population(n_exc+n_inh, sim.IF_cond_exp, cell_params, label="All cells")
    cells = {
        'excitatory': all_cells[:n_exc],
        'inhibitory': all_cells[n_exc:],
        'input': sim.Population(n_input, sim.SpikeSourcePoisson, stimulation_params, label="Input")
    }
    
    rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
    uniform_distr = RandomDistribution(
                        'uniform',
                        [cell_params['v_reset'], cell_params['v_thresh']],
                        rng=rng)
    all_cells.initialize('v', uniform_distr)
    
    connections = {}
    for name, pconn, target in (
        ('excitatory', pconn_recurr, 'excitatory'),
        ('inhibitory', pconn_recurr, 'inhibitory'),
        ('input',      pconn_input,  'excitatory'),
    ):
        connector = sim.FixedProbabilityConnector(pconn, weights=weights[name], delays=delay)
        connections[name] = sim.Projection(cells[name], all_cells, connector,
                                           target=target, label=name, rng=rng)
    
    all_cells.record()
    cells['excitatory'][0:2].record_v()
    
    sim.run(tstop)
    
    E_count = cells['excitatory'].meanSpikeCount()
    I_count = cells['inhibitory'].meanSpikeCount()
    
    print "Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,)
    print "Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,)
    
    sim.end()

@register
def scenario2(sim):
    """
    Array of neurons, each injected with a different current.
    
    firing period of a IF neuron injected with a current I:
    
    T = tau_m*log(I*tau_m/(I*tau_m - v_thresh*cm))
    
    (if v_rest = v_reset = 0.0)

    we set the refractory period to be very large, so each neuron fires only
    once (except neuron[0], which never reaches threshold).
    """
    set_simulator(sim)
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
    neurons = sim.Population(n, sim.IF_curr_exp, cell_params)
    neurons.initialize('v', 0.0)
    I = numpy.arange(I0, I0+1.0, 1.0/n)
    currents = [sim.DCSource(start=t_start, stop=t_start+duration, amplitude=amp)
                  for amp in I]
    for j, (neuron, current) in enumerate(zip(neurons, currents)):
        if j%2 == 0:                      # these should
            neuron.inject(current)        # be entirely
        else:                             # equivalent
            current.inject_into([neuron])
    neurons.record_v()
    neurons.record()
    
    sim.run(t_stop)
    
    spikes = neurons.getSpikes()
    assert_equal(spikes.shape, (99,2)) # first cell does not fire
    spikes = sort_by_column(spikes, 0)
    spike_times = spikes[:,1]
    expected_spike_times = t_start + tau_m*numpy.log(I*tau_m/(I*tau_m - v_thresh*cm))
    a = spike_times = spikes[:,1]
    b = expected_spike_times[1:]
    max_error = abs((a-b)/b).max()
    print "max error =", max_error
    assert max_error < 0.005, max_error
    #neurons.printSpikes("scenario2_%s.spikes" % sim.__name__)
    sim.end()
    #return a,b, spikes
    
@register
def ticket166(sim):
    """
    Check that changing the spike_times of a SpikeSourceArray mid-simulation
    works (see http://neuralensemble.org/trac/PyNN/ticket/166)
    """
    dt = 0.1 # ms
    t_step = 100.0 # ms
    lag = 3.0 # ms
    interactive = True
    
    if interactive:
        import pylab
        pylab.rcParams['interactive'] = interactive
    
    set_simulator(sim)
    sim.setup(timestep=dt)
    
    spikesources = sim.Population(2, sim.SpikeSourceArray)
    cells = sim.Population(2, sim.IF_curr_exp)
    conn = sim.Projection(spikesources, cells, sim.OneToOneConnector(weights=0.1))
    cells.record_v()
    
    spiketimes = numpy.arange(2.0, t_step, t_step/13.0)
    spikesources[0].spike_times = spiketimes
    spikesources[1].spike_times = spiketimes + lag
    
    t = sim.run(t_step) # both neurons depolarized by synaptic input
    t = sim.run(t_step) # no more synaptic input, neurons decay
    
    spiketimes += 2*t_step
    spikesources[0].spike_times = spiketimes
    # note we add no new spikes to the second source
    t = sim.run(t_step) # first neuron gets depolarized again
    
    final_v_0 = cells[0:1].get_v()[-1,2]
    final_v_1 = cells[1:2].get_v()[-1,2]
    
    sim.end()
    
    if interactive:
        id, t, vtrace = cells[0:1].get_v().T
        print vtrace.shape
        print t.shape
        pylab.plot(t, vtrace)
        id, t, vtrace = cells[1:2].get_v().T
        pylab.plot(t, vtrace)
    
    assert final_v_0 > -64.0  # first neuron has been depolarized again
    assert final_v_1 < -64.99 # second neuron has decayed back towards rest
    
@register
def test_reset(sim):
    """
    Run the same simulation n times without recreating the network,
    and check the results are the same each time.
    """
    repeats = 3
    sim.setup()
    p = sim.Population(1, sim.IF_cond_exp, {"i_offset": 0.1})
    p.record_v()
    
    data = []
    for i in range(repeats):
        sim.run(10.0)
        data.append(p.get_v())
        sim.reset()
        
    sim.end()

    for rec in data:
        assert_arrays_almost_equal(rec, data[0], 1e-12)

@register
def test_setup(sim):
    """
    Run the same simulation n times, recreating the network each time,
    and check the results are the same each time.
    """
    n = 3
    data = []

    for i in range(n):
        sim.setup()
        p = sim.Population(1, sim.IF_cond_exp, {"i_offset": 0.1})
        p.record_v()
        sim.run(10.0)
        data.append(p.get_v())
        sim.end()

    assert len(data) == n
    assert data[0].size > 0
    for rec in data:
        assert_arrays_equal(rec, data[0])