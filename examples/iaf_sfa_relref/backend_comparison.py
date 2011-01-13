import NeuroTools.parameters
from NeuroTools import stgen
import pyNN.neuron as neuron
import pyNN.nest as nest

pyNN_backends = {'nest':nest,'neuron':neuron}

def run(sim):
    """ Run the mcb simulation with known random numbers for the given simulator backend"""

    stg = stgen.StGen()
    stg.seed(12345)

    params = NeuroTools.parameters.ParameterSet('standard_neurons.yaml')
    myModel = sim.IF_cond_exp_gsfa_grr
    popE = sim.Population((1,),myModel,params.excitatory,label='popE')
    popI = sim.Population((1,),myModel,params.inhibitory,label='popI')

    rateE = 6.0 # firing rate of ghost excitatory neurons
    rateI = 10.5 # firing rate of ghost inhibitoryneurons
    connectionsE = 1000.0 # number of E connections every neuron recieves
    connectionsI = 250.0 # number of I connections every neuron recieves
    tsim = 100000.0 # ms

    globalWeight = 0.002 # Weights of all connection in uS
    dt = 0.1 # simulation time step in milliseconds

    sim.setup(timestep=dt, min_delay=dt, max_delay=30.0, debug=True, quit_on_end=False)

    #poissonE_params = {'rate': rateE*connectionsE, 'start': 0.0, 'duration': tsim}
    #poissonE_params = {'rate': rateE, 'start': 0.0, 'duration': tsim}
    #poissonI_params = {'rate': rateI*connectionsI, 'start': 0.0, 'duration': tsim}
    #poissonI_params = {'rate': rateI, 'start': 0.0, 'duration': tsim}

    spike_times_E = stg.poisson_generator(rateE*connectionsE, 0.0, tsim, array=True)
    spike_times_I = stg.poisson_generator(rateI*connectionsI, 0.0, tsim, array=True)

    poissonE = sim.Population((1,),cellclass=sim.SpikeSourceArray,
                              cellparams={'spike_times':spike_times_E},label='poissonE')

    poissonI = sim.Population((1,),cellclass=sim.SpikeSourceArray,
                              cellparams={'spike_times':spike_times_I},label='poissonI')

    myconn = sim.AllToAllConnector(weights=globalWeight, delays=dt)

    prjE_E = sim.Projection(poissonE, popE, method=myconn, target='excitatory')
    prjI_E = sim.Projection(poissonI, popE, method=myconn, target='inhibitory')

    prjE_I = sim.Projection(poissonE, popI, method=myconn, target='excitatory')
    prjI_I = sim.Projection(poissonI, popI, method=myconn, target='inhibitory')

    ## Record the spikes ##
    popE.record(to_file=False)
    popE.record_v(to_file=False)
    popI.record(to_file=False)
    popI.record_v(to_file=False)

    sim.run(tsim)

    ## Get spikes ##
    spikesE = popE.getSpikes()
    v_E = popE.get_v()
    spikesI = popI.getSpikes()
    v_I = popE.get_v()
                              
    #print spikesE
    # should be about 6.0Hz
    print float(len(spikesE))/tsim*1000.0
    # should be about 10.0Hz
    print float(len(spikesI))/tsim*1000.0

    sim.end()

    return spikesE, spikesI, v_E, v_I


results = {}
for sim_name, sim in pyNN_backends.iteritems():
    results[sim_name] = run(sim)
    
