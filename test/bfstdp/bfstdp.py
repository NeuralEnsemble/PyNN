"""
Learning basis functions to implement functions of one population-encoded
variable using STDP.

The model has an Input Layer (cellLayer[0]) and a Training Layer
(cellLayer[1]), each consisting of spike sources, and projecting to an Output
Layer (cellLayer[2]) consisting of integrate-and-fire neurons.

The synaptic weights from Training-->Output are fixed.
The synaptic weights from Input-->Output are plastic and obey a STDP rule.

During training, the Input Layer receives input x, and the Training Layer
input f(x). After training, the Training Layer is silent, and an input x to
the Input Layer produces an output f(x) in the Output Layer.

Uses the NetStimVR2 mechanism, rather than VecStimMs

Andrew P. Davison, UNIC, CNRS
Original hoc version: July 2004-May 2006
Python version: February 2008
"""

import sys
import datetime
import pyNN.neuron as sim
import pyNN.random as random
from numpy import exp, cos, pi
import numpy

sim.Timer.start()

#xopen("plotweights.hoc")

# =-=-= Global Parameters =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

seed             = 0           # Seed for the random number generator
ncells           = 30          # Number of input spike trains per layer
pconnect         = 1.0         # Connection probability
wmax             = 0.02        # Maximum synaptic weight
f_winhib         = 0.0         # Inhibitory weight = f_winhib*wmax (fixed)
f_wtr            = 1.0         # Max training weight = f_wtr*wmax
min_delay        = 1e-13        # Non-zero minimum synaptic delay
syndelay         = 0           # Synaptic delay relative to min_delay
tauLTP_StdwaSA   = 20          # (ms) Time constant for LTP
tauLTD_StdwaSA   = 20          # (ms) Time constant for LTD
B                = 1.06        # B = (aLTD*tauLTD)/(aLTP*tau_LTP)
aLTP             = 0.01        # Amplitude parameter for LTP
Rmax             = 60          # (Hz) Peak firing rate of input distribution
Rmin             = 0           # (Hz) Minumum input firing rate
Rsigma           = 0.2         # Width parameter for input distribution
alpha            = 1.0         # Gain of Training Layer rates compared to Input Layer
correlation_time = 20          # (ms) 
bgRate           = 1000        # (Hz) Firing rate for background activity
bgWeight         = 0.02        # Weight for background activity
funcstr          = "sin"       # Label for function to be approximated
nfuncparam       = 1           # Number of parameters of function
k                = [0.0]       # Function parameter(s)
wtr_square       = True        # Sets square or bell-shaped profile for T-->O weights
wtr_sigma        = 0.15        # Width parameter for Training-->Output weights
noise            = 1           # Noise parameter
histbins         = 100         # Number of bins for weight histograms
record_spikes    = True        # Whether or not to record spikes
wfromfile        = False       # if positive, read connections/weights from file
infile           = ""          # File to read connections/weights from
tstop            = 1e7         # (ms)
trw              = 1e6         # (ms) Time between reading input spikes/printing weights
numhist          = 10          # Number of histograms between each weight printout
label            = "bfstdp_demo_" # Extra label for labelling output files
datadir          = ""          # Sub-directory of Data for writing output files
tau_m            = 20          # Membrane time constant

# =-=-= Create utility objects  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

rng = random.NumpyRNG(seed)

def set_training_weights(prj):
    # Set the Training-->Output weights
    # what about setWeights() with a 2D array? weight values for non-existent connections would be ignored?
    global ncells
    if wtr_square:
        weights = numpy.fromfunction(lambda i,j: 1.0*(abs(i-j)>ncells*(1-wtr_sigma)), (ncells, ncells)) + \
                  numpy.fromfunction(lambda i,j: 1.0*(abs(i-j)<ncells*wtr_sigma), (ncells, ncells))
    else:
        weights = numpy.fromfunction(lambda i,j: exp( (cos(2*pi*(i-j)/ncells) - 1) / (wtr_sigma*wtr_sigma) ),
                                     (ncells, ncells))
    weights *= f_wtr*wmax
    prj.setWeights(weights.flatten())

# =-=-= Create the network  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

sim.setup(min_delay=min_delay, debug=True, use_cvode=True)
#sim.hoc_execute(['nrn_load_dll("%s/i686/.libs/libnrnmech.so")' % os.getcwd()])
sim.hoc_execute(['xopen("spikeSourceVR.hoc")',
                 'xopen("intfire4nc.hoc")'])
# Input spike trains are implemented using NetStimVR2s.
print "Creating network layers (time %g s)" % sim.Timer.elapsedTime()

TRANSFORMATIONS = {
    "":     0,
    "mul":  1,
    "sin":  2,
    "sq":   3,
    "asin": 4, 
    "sinn": 5
}

cellParams = [None, None, None]
cellParams[0] = {'noise': 1, 'transform': 0, 'prmtr': 0,
                 'tau_corr': correlation_time, 'seed': seed}
cellParams[1] = {'noise': 1, 'transform': TRANSFORMATIONS[funcstr],
                 'prmtr': k[0], 'alpha': alpha}
cellParams[2] = {
    'taum': tau_m,
    'taue': 5,
    'taui1': 10,
    'taui2': 15
}

cellLayer = [None, None, None]
# Create network layers
for layer in 0,1:
    cellLayer[layer] = sim.Population(ncells, "SpikeSourceVariableRate", cellParams[layer])
    for i in range(ncells):
        cellLayer[layer].cell[i].theta = float(i)/ncells
    
hoc_commands = ["Rmax_NetStimVR2 = %g" % Rmax,
                "Rmin_NetStimVR2 = %g" % Rmin,
                "sigma_NetStimVR2 = %g" % Rsigma]
sim.hoc_execute(hoc_commands)

cellLayer[2] = sim.Population(ncells, "IntFire4nc", cellParams[2])

# Create synaptic connections
print "Creating synaptic connections (time %g s)" % sim.Timer.elapsedTime()

initial_weight_distribution = random.RandomDistribution('uniform', (0,wmax), rng)

# Turn on STDP for Input-->Output connections
print "  Defining STDP configuration for Input-->Output connections"
aLTD = B*aLTP*tauLTP_StdwaSA/tauLTD_StdwaSA
stdp_model = sim.STDPMechanism(timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
                               weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=wmax,
                                                                              A_plus=aLTP,
                                                                              A_minus=aLTD))
synapse_dynamics = sim.SynapseDynamics(fast=None, slow=stdp_model)

conn = [None, None, None]
if wfromfile: # read connections from file
    conn[0] = sim.Projection(cellLayer[0], cellLayer[2],
                             method=sim.FromFileConnector("%s.conn1.conn" % infile),
                             source="",
                             target="syn",
                             synapse_dynamics=synapse_dynamics)
    conn[1] = sim.Projection(cellLayer[1], cellLayer[2],
                             method=sim.FromFileConnector("%s.conn2.conn" % infile),
                            source="",
                            target="syn",
                            synapse_dynamics=None)
    if f_winhib != 0:
      filename = "%s.conn2.conn" % infile
      conn[2] = sim.Projection(cellLayer[2], cellLayer[2],
                               method=sim.FromFileConnector(filename),
                               source="syn",
                               target="syn")
else:         # or generate them according to the rules specified
    connector = sim.FixedProbabilityConnector(pconnect) # can you reuse a connector?
    conn[0] = sim.Projection(cellLayer[0], cellLayer[2],
                             method=connector,
                             source="syn",
                             target="syn",
                             rng=rng,
                             synapse_dynamics=synapse_dynamics)
    conn[0].randomizeWeights(initial_weight_distribution)
    conn[1] = sim.Projection(cellLayer[1], cellLayer[2],
                             method=connector,
                             source="syn",
                             target="syn",
                             rng=rng,
                             synapse_dynamics=None)
    set_training_weights(conn[1])
    if syndelay < 0:
      conn[0].setDelays(min_delay + -1*syndelay)
      conn[1].setDelays(min_delay)
    elif syndelay > 0:
        conn[0].setDelays(min_delay)
        conn[1].setDelays(min_delay + syndelay)

    if f_winhib != 0:
        conn[2] = Population(cellLayer[2], cellLayer[2],
                             method=sim.AllToAllConnector(allow_self_connections=False),
                             source="syn",
                             target="syn")
        conn[2].setWeights(wmax*f_winhib)

# Set background input
background_input = sim.Population(cellLayer[2].dim, sim.SpikeSourcePoisson, {'rate': bgRate,
                                                                             'duration': tstop})
background_connect = sim.Projection(background_input, cellLayer[2],
                                    method=sim.OneToOneConnector(weights=bgWeight),
                                    target="syn")

# Turn on recording of spikes
if record_spikes:
    cellLayer[0].record()
    cellLayer[1].record()  
    cellLayer[2].record()
#cellLayer[2].record_v(record_from=2)

# =-=-= Procedures =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Utility procedures ----------------------------------------------------------

def set_fileroot():
    global datadir, label, funcstr, fileroot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fileroot = "Data/%s/%s%s" % (datadir, label, funcstr)
    for i in range(nfuncparam):
        fileroot += "-%3.1f" % k[i]
    fileroot += "_%s" % timestamp

# Procedures to set weights ---------------------------------------------------



# Procedures for writing results to file --------------------------------------

def save_parameters():
    global fileroot # actually all the parameters are global as well, but I can't be bothered to list them
    filename = "%s.param" % fileroot
    f = open(filename, 'w')
    lines = ["# Parameters for bfstdp.py"]
    lines += ["%-17s = %d" % (p, eval(p)) for p in ("seed", "ncells", "nfuncparam")]
    lines += ["%-17s = %f" % (p, eval(p)) for p in (
        "pconnect", "wmax", "f_winhib", "f_wtr", "syndelay", "tauLTP_StdwaSA",
        "tauLTD_StdwaSA", "B","aLTP", "Rmax", "Rmin", "Rsigma", "alpha",
        "correlation_time", "bgWeight", "bgRate", "wtr_sigma",
        "noise", "tau_m")]
    lines += ["%-17s = %s" % (p, eval(p)) for p in ("wtr_square", "k")]
    lines += ['%-17s = "%s"' % (p, eval(p)) for p in ("funcstr",)]
    if wfromfile:
        lines += ['%-17s = "%s"' % (p, eval(p)) for p in ("infile",)]
    f.write("\n".join(lines))
    f.close()

def print_rasters():
    global fileroot
    for i in 0,1,2:
        cellLayer[i].printSpikes("%s.cell%d.ras" % (fileroot,i+1))

def print_weights(projection_id):
    global fileroot
    conn[projection_id].printWeights1("%s.conn%d.w" % (fileroot, projection_id+1))

def save_connections():
     for i in range(3-(f_winhib==0)):
       conn[i].saveConnections("%s.conn%d.conn" % (fileroot,i+1))

def print_weight_distribution(histfileobj):
    # Pointless to calculate distribution for inhibitory weights (i=1,2)
    hist, bins = conn[0].weightHistogram(min=0, max=wmax, nbins=histbins)
    fmt = "%g "*len(hist) + "\n"
    histfileobj.write(fmt % tuple(hist))

#def print_delta_t(binwidth, range, normalize):
#  histbins = 2*range+1
#  deltat_hist = new Vector(histbins)
#  for layer = 0,1 {
#    total_size = deltat_vec[layer][0].size() + deltat_vec[layer][1].size() + deltat_vec[layer][2].size()
#    for ii = 0,2 {
#      deltat_hist.hist(deltat_vec[layer][ii],-range-0.5,histbins,binwidth)
#      if normalize: deltat_hist.div(total_size)
#      sprint(filename,"%s.conn%d.deltat%d",fileroot,layer+1,ii)
#      fileobj.wopen(filename)
#      for i = 0, histbins-1 { #print in a column
#	fileobj.printf("%g\t%g\n",-range+binwidth*i,deltat_hist.x[i])
#      }
#      #deltat_vec.printf(fileobj)
#      fileobj.close()
#    }
#  }
#}

# Procedures that process recorded data ---------------------------------------

#def calc_delta_t(binwidth, range, normalize=False):
#  # Calculate the distribution of spike-time differences (post-pre)
#  # in three classes: connections for which d < 0.1, d < 0.2, d >= 0.2
#  if (record_spikes) {
#    for ii = 0,2 {
#      for layer = 0,1 {
#	deltat_vec[layer][ii] = new Vector(1e6)
#	m[layer][ii] = 0
#      }
#    }
#    for i = 0,ncells-1 {
#      nspikes_post = cellLayer[2].cell[i].spiketimes.size()
#      if (nspikes_post > 0) {
#	for j = 0, nspikes_post-1 {
#	  tpost = cellLayer[2].cell[i].spiketimes.x[j]
#	  for k = 0,ncells-1 {
#	    for layer = 0,1 {
#	      if (layer==0) {
#		d  = i/ncells - (sin(2*PI*k/ncells)+1)/2
#	      } else {
#		d = i/ncells - k/ncells
#	      }
#	      if (d < -0.5) d += 1
#	      if (d >= 0.5) d -= 1
#	      d = abs(d)
#	      if (d < 0.1) {
#		ii = 0
#	      } else {
#		if (d < 0.2) {
#		  ii = 1
#		} else {
#		  ii = 2
#		}
#	      }
#	      nspikes_pre = spikerec[layer].x[k].size()
#	      if (nspikes_pre > 0) {
#		for l = 0, nspikes_pre-1 {
#		  deltat = tpost - spikerec[layer].x[k].x[l]
#		  if (deltat < range && deltat > -1*range) {
#		    deltat_vec[layer][ii].x[m[layer][ii]] = deltat
#		    m[layer][ii] += 1
#		    if (m[layer][ii] >= deltat_vec[layer][ii].size()-1) {
#		      deltat_vec[layer][ii].resize(2*deltat_vec[layer][ii].size)
#		      printf("deltat_vec[%d][%d] resized\n",layer,ii)
#		    }
#		  }
#		}
#	      }
#	    }
#	  }
#	}
#      }
#    }
#    printf("Spike pairs: %d,%d  %d,%d  %d,%d\n",m[0][0],m[1][0],m[0][1],m[1][1],m[0][2],m[1][2])
#    for ii = 0,2 {
#      deltat_vec[0][ii].resize(m[0][ii])
#      deltat_vec[1][ii].resize(m[1][ii])
#    }
#    print_delta_t(binwidth, range, normalize)
#    
#  }
#}



# Procedures that run simulations ---------------------------------------------

def run_training():
    """
    Training the network. The weight histogram is written to
    file every trw ms. The weights are written to file every
    thist = trw/numhist ms. The spike-times of the network
    cells are written to file at the end.
    """
    global fileroot
    
    on_StdwaSA = 1
    thist = int(trw/numhist)
    
    histfileobj = open("%s.conn1.whist" % fileroot, "w")
     
    save_parameters()
    save_fileroot = fileroot
    fileroot = "%s_%d" % (save_fileroot,0)
    print_weights(0)
    print_weights(1)
    save_connections()
    
    i = 0
    j = 0
    
    #running_ = 1
    #setup_weight_plot()
    #finitialize(-65)
    #plot_weights(conn[0])
    #starttime = startsw()
    sim.Timer.reset()
    t = 0
    while t < tstop:
        fileroot = "%s_%d" % (save_fileroot, j*thist)
        print_weight_distribution(histfileobj)
        if i == numhist:
            print_weights(0)
            i = 0
            print "--- Simulated %d seconds in %d seconds\r" % (int(t/1000), sim.Timer.elapsedTime()),
            sys.stdout.flush()
        i += 1
        j += 1
        t = sim.run(thist)
        #plot_weights(conn[0])
    
    print "--- Simulated %d seconds in %d seconds\n" % (int(t/1000), sim.Timer.elapsedTime())
    
    fileroot = "%s_%d" % (save_fileroot, j*thist)
    print_weights(0)
    print_weights(1) # for debugging. Should not have changed since t = 0
    print_weight_distribution(histfileobj)
    save_connections()
    
    fileroot = save_fileroot
    
    # This corrects the pre-synaptic spiketimes for syndelay.
    # This is necessary because nc.record records spike times at the source
    # whereas we want to know them at the target.
    
    #if (syndelay < 0) {
    #  for i = 0,ncells-1 {
    #    spikerec[0].x[i].add(-1*syndelay)
    #  } 
    #} else if (syndelay > 0) {
    #  for i = 0,ncells-1 {
    #    spikerec[1].x[i].add(syndelay)
    #  }
    #}
    
    print_rasters()
    
    histfileobj.close()
    print "Training complete. Time ", sim.Timer.elapsedTime()
    #calc_delta_t(1.0,1000,0)
  
    
# =-=-= Initialize the network =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

set_fileroot()
sim.h.cvode.active(1)
sim.h.cvode.use_local_dt(1)         # The variable time step method must be used.
sim.h.cvode.condition_order(2)      # Improves threshold-detection.

print "Finished set-up (time %g s)" % sim.Timer.elapsedTime()

print "Running training ..."

run_training()

#sim.h('for i = 0, population0.count()-1 {population0.o(i).spiketimes.printf()}')
