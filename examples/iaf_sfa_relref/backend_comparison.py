import NeuroTools.parameters
from NeuroTools import stgen
import pyNN.neuron as neuron
import pyNN.nest as nest

pyNN_backends = {'nest': nest, 'neuron': neuron}


def run(sim):
    """ Run the mcb simulation with known random numbers for the given simulator backend"""

    stg = stgen.StGen()
    stg.seed(12345)

    rateE = 6.0  # firing rate of ghost excitatory neurons
    rateI = 10.5  # firing rate of ghost inhibitoryneurons
    connectionsE = 1000.0  # number of E connections every neuron recieves
    connectionsI = 250.0  # number of I connections every neuron recieves
    tsim = 1000.0  # ms

    globalWeight = 0.002  # Weights of all connection in uS
    dt = 0.01  # simulation time step in milliseconds

    sim.setup(timestep=dt, min_delay=dt, max_delay=30.0, debug=True, quit_on_end=False)

    params = NeuroTools.parameters.ParameterSet('standard_neurons.yaml')
    myModel = sim.IF_cond_exp_gsfa_grr
    popE = sim.Population((1,), myModel, params.excitatory, label='popE')
    popI = sim.Population((1,), myModel, params.inhibitory, label='popI')

    #poissonE_params = {'rate': rateE*connectionsE, 'start': 0.0, 'duration': tsim}
    #poissonE_params = {'rate': rateE, 'start': 0.0, 'duration': tsim}
    #poissonI_params = {'rate': rateI*connectionsI, 'start': 0.0, 'duration': tsim}
    #poissonI_params = {'rate': rateI, 'start': 0.0, 'duration': tsim}

    spike_times_E = stg.poisson_generator(rateE * connectionsE, 0.0, tsim, array=True)
    spike_times_I = stg.poisson_generator(rateI * connectionsI, 0.0, tsim, array=True)

    poissonE = sim.Population((1,), cellclass=sim.SpikeSourceArray,
                              cellparams={'spike_times': spike_times_E}, label='poissonE')

    poissonI = sim.Population((1,), cellclass=sim.SpikeSourceArray,
                              cellparams={'spike_times': spike_times_I}, label='poissonI')

    myconn = sim.AllToAllConnector(weights=globalWeight, delays=dt)

    prjE_E = sim.Projection(poissonE, popE, method=myconn, target='excitatory')
    prjI_E = sim.Projection(poissonI, popE, method=myconn, target='inhibitory')

    prjE_I = sim.Projection(poissonE, popI, method=myconn, target='excitatory')
    prjI_I = sim.Projection(poissonI, popI, method=myconn, target='inhibitory')

    ## Record the spikes ##
    popE.record(to_file=False)
    popE.record_v(to_file=False)
    popE.record_gsyn(to_file=False)
    popI.record(to_file=False)
    popI.record_v(to_file=False)
    popI.record_gsyn(to_file=False)
    #popE.record()
    #popE.record_v()
    #popI.record()
    #popI.record_v()

    sim.run(tsim)

    ## Get spikes ##
    spikesE = popE.getSpikes()
    v_E = popE.get_v()
    g_E = popE.get_gsyn()
    spikesI = popI.getSpikes()
    v_I = popE.get_v()
    g_I = popI.get_gsyn()

    #print(spikesE)
    # should be about 6.0Hz
    print(float(len(spikesE)) / tsim * 1000.0)
    # should be about 10.0Hz
    print(float(len(spikesI)) / tsim * 1000.0)

    sim.end()

    return spikesE, spikesI, v_E, v_I, g_E, g_I


results = {}
for sim_name, sim in pyNN_backends.items():
    print(sim_name)
    results[sim_name] = run(sim)
    #v_E = results[sim_name][2]
    #plot(v_E[:,1],v_E[:,2], label=sim_name)

    g_E = results[sim_name][4]
    plot(g_E[:, 1], g_E[:, 2], label=sim_name)

legend()


