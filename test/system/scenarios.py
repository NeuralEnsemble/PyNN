from pyNN.random import NumpyRNG, RandomDistribution
from pyNN import common, recording

def set_simulator(sim):
    common.simulator = sim.simulator
    recording.simulator = sim.simulator

def scenario1(sim):
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


    