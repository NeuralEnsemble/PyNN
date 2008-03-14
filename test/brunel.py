"""

Conversion of the Brunel network implemented in nest-1.0.13/examples/brunel.sli
to use PyNN

Andrew Davison, UNIC, CNRS
May 2006

$Id$

"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "oldneuron"    # run using nrngui -python
exec("from pyNN.%s import *" % simulator)

from pyNN.random import NumpyRNG, RandomDistribution
Timer = Timer()

# === Define parameters ========================================================

downscale   = 50      # scale number of neurons down by this factor
                      # scale synaptic weights up by this factor to
                      # obtain similar dynamics independent of size
order       = 50000   # determines size of network:
                      # 4*order excitatory neurons
                      # 1*order inhibitory neurons
Nrec        = 50      # number of neurons to record from, per population
epsilon     = 0.1     # connectivity: proportion of neurons each neuron projects to
    
# Parameters determining model dynamics, cf Brunel (2000), Figs 7, 8 and Table 1
# here: Case C, asynchronous irregular firing, ~35 Hz
eta         = 2.0     # rel rate of external input
g           = 5.0     # rel strength of inhibitory synapses
J           = 0.1     # synaptic weight [mV]
delay       = 1.5     # synaptic delay, all connections [ms]  

# single neuron parameters
tauMem      = 20.0    # neuron membrane time constant [ms]
tauSyn      = 0.1     # synaptic time constant [ms]
tauRef      = 2.0     # refractory time [ms]
U0          = 0.0     # resting potential [mV]
theta       = 20.0    # threshold 

# simulation-related parameters  
simtime     = 100.0  # simulation time [ms] 
dt          = 0.1     # simulation step length [ms]   

# seed for random generator used when building connections
connectseed = 12345789   
use_RandomArray = True # use Python rng rather than NEST rng

# seed for random generator(s) used during simulation
kernelseed  = 43210987      

exfilename  = "brunel_ex_%s.ras" % simulator # output file for excit. population  
infilename  = "brunel_in_%s.ras" % simulator # output file for inhib. population  
vexfilename = "brunel_ex_%s.v" % simulator # output file for membrane potential traces
vinfilename = "brunel_in_%s.v" % simulator # output file for membrane potential traces
  
# === Calculate derived parameters =============================================

# scaling: compute effective order and synaptic strength
order_eff = int(float(order)/downscale)
J_eff     = J*downscale
  
# compute neuron numbers
NE = int(4*order_eff)  # number of excitatory neurons
NI = int(1*order_eff)  # number of inhibitory neurons
N  = NI + NE           # total number of neurons

# compute synapse numbers
CE   = int(epsilon*NE)  # number of excitatory synapses on neuron
CI   = int(epsilon*NI)  # number of inhibitory synapses on neuron
C    = CE + CI          # total number of internal synapses per n.   
Cext = CE               # number of external synapses on neuron  

# synaptic weights, scaled for alpha functions, such that
# for constant membrane potential, charge J would be deposited
fudge = 0.00041363506632638 # ensures dV = J at V=0  
  
# excitatory weight: JE = J_eff / tauSyn * fudge
JE = (J_eff/tauSyn)*fudge 
  
# inhibitory weight: JI = - g * JE
JI = -g*JE  
  
# threshold, external, and Poisson generator rates:
nu_thresh = theta/(J_eff*CE*tauMem)
nu_ext    = eta*nu_thresh     # external rate per synapse
p_rate    = 1000*nu_ext*Cext  # external input rate per neuron (Hz)
                                    
# number of synapses---just so we know
Nsyn = (C+1)*N + 2*Nrec  # number of neurons * (internal synapses + 1 synapse from PoissonGenerator) + 2synapses" to spike detectors

# put cell parameters into a dict
cell_params = {'tau_m'      : tauMem,
               'tau_syn_E'  : tauSyn,
               'tau_syn_I'  : tauSyn,
               'tau_refrac' : tauRef,
               'v_rest'     : U0,
               'v_reset'    : U0,
               'v_thresh'   : theta,
               'cm'         : 0.001}     # (nF)

# === Build the network ========================================================

# clear all existing network elements and set resolution and limits on delays.
# For NEST, limits must be set BEFORE connecting any elements

extra = {'threads' : 2}

myid = setup(timestep=dt, max_delay=delay, **extra)

if extra.has_key('threads'):
    print "%d Initialising the simulator with %d threads..." %(myid, extra['threads'])
else:
    print "%d Initialising the simulator with single thread..." %(node_id)

# Small function to display information only on node 1
def nprint(s):
    if (myid == 0):
        print s

Timer.start() # start timer on construction    

print "%d Setting up random number generator" %myid
rng = NumpyRNG(kernelseed+myid)

print "%d Creating excitatory population." %myid
E_net = Population((NE,),IF_curr_alpha,cell_params,"E_net")

print "%d Creating inhibitory population." %myid
I_net = Population((NI,),IF_curr_alpha,cell_params,"I_net")

print "%d Initialising membrane potential to random values." %myid
rng2 = NumpyRNG(kernelseed+myid)
uniformDistr = RandomDistribution('uniform',[U0,theta],rng2)
E_net.randomInit(uniformDistr)
I_net.randomInit(uniformDistr)

print "%d Creating excitatory Poisson generator." %myid
expoisson = Population((NE,), SpikeSourcePoisson,{'rate': p_rate},"expoisson")

print "%d Creating inhibitory Poisson generator." %myid
inpoisson = Population((NI,), SpikeSourcePoisson,{'rate': p_rate},"inpoisson")

# Record spikes
print "%d Setting up recording in excitatory population." %myid
E_net.record(Nrec)
E_net.record_v([E_net[0],E_net[1]])

print "%d Setting up recording in inhibitory population." %myid
I_net.record(Nrec)
I_net.record_v([I_net[0],I_net[1]])

E_Connector = FixedProbabilityConnector(epsilon, weights = JE, delays = delay)
I_Connector = FixedProbabilityConnector(epsilon, weights = JI, delays = delay)
ext_Connector = OneToOneConnector(weights = JE, delays = dt)

print "%d Connecting excitatory population."  %myid
E_to_E = Projection(E_net, E_net, E_Connector, rng=rng, target="excitatory")
print "E --> E\t\t", len(E_to_E), "connections"
I_to_E = Projection(I_net, E_net, I_Connector, rng=rng, target="inhibitory")
print "I --> E\t\t", len(I_to_E), "connections"
input_to_E = Projection(expoisson, E_net, ext_Connector, target="excitatory")
print "input --> E\t", len(input_to_E), "connections"

print "%d Connecting inhibitory population." %myid
E_to_I = Projection(E_net, I_net, E_Connector, rng=rng, target="excitatory")
print "E --> I\t\t", len(E_to_I), "connections"
I_to_I = Projection(I_net, I_net, I_Connector, rng=rng, target="inhibitory")
print "I --> I\t\t", len(I_to_I), "connections"
input_to_I = Projection(inpoisson, I_net, ext_Connector, target="excitatory")
print "input --> I\t", len(input_to_I), "connections"

# read out time used for building
buildCPUTime = Timer.elapsedTime()
# === Run simulation ===========================================================

# run, measure computer time
Timer.start() # start timer on construction
print "%d Running simulation." %myid
run(simtime)
simCPUTime = Timer.elapsedTime()

# write data to file
E_net.printSpikes(exfilename)
I_net.printSpikes(infilename)
E_net.print_v(vexfilename)
I_net.print_v(vinfilename)

E_rate = E_net.meanSpikeCount()*1000.0/simtime
I_rate = I_net.meanSpikeCount()*1000.0/simtime

# write a short report
nprint("\n--- Brunel Network Simulation ---")
nprint("Number of Neurons  : %d" %N)
nprint("Number of Synapses : %d" %Nsyn)
nprint("Input firing rate  : %g" %p_rate)
nprint("Excitatory weight  : %g" %JE)
nprint("Inhibitory weight  : %g" %JI)
nprint("Excitatory rate    : %g Hz" %E_rate)
nprint("Inhibitory rate    : %g Hz" %I_rate)
nprint("Build time         : %g s" %buildCPUTime)   
nprint("Simulation time    : %g s" %simCPUTime)

  
# === Clean up and quit ========================================================

end()

