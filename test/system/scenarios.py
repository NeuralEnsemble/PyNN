from pyNN.random import NumpyRNG, RandomDistribution
from pyNN import common, recording
from nose.tools import assert_equal
import numpy
from pyNN.utility import assert_arrays_equal, sort_by_column

def set_simulator(sim):
    common.simulator = sim.simulator
    recording.simulator = sim.simulator

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


def scenario2(sim):
    """
    Array of neurons, each injected with a different current.
    
    firing period of a IF neuron injected with a current I:
    (where v_rest = v_reset = 0.0)
    T = tau_m*log(I*tau_m/(I*tau_m - v_thresh*cm))
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
    spikes = sort_by_column(spikes, 0)
    assert_equal(spikes.shape, (99,2)) # first cell does not fire
    spike_times = spikes[:,1]
    expected_spike_times = t_start + tau_m*numpy.log(I*tau_m/(I*tau_m - v_thresh*cm))
    a = spike_times = spikes[:,1]
    b = expected_spike_times[1:]
    max_error = abs((a-b)/b).max()
    assert max_error < 0.005, max_error
    #neurons.printSpikes("scenario2_%s.spikes" % sim.__name__)
    sim.end()
    #return a,b, spikes