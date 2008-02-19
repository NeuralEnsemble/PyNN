# encoding: utf-8
"""
Learning basis functions to implement functions of one population-encoded
variable using STDP.

The model has an Input Layer and a Training Layer, each consisting of spike
sources, and projecting to an Output Layer consisting of integrate-and-fire
neurons.

The synaptic weights from Training-->Output are fixed.
The synaptic weights from Input-->Output are plastic and obey an STDP rule.

During training, the Input Layer receives input x, and the Training Layer
input f(x). After training, the Training Layer is silent, and an input x to
the Input Layer produces an output f(x) in the Output Layer.

For a reference, see:
  Davison A.P. and Frégnac Y. (2006) Learning crossmodal spatial transformations
  through spike-timing-dependent plasticity. J. Neurosci 26: 5604-5615.

Based on an original NEURON model, see:
  http://senselab.med.yale.edu/senselab/ModelDB/ShowModel.asp?model=64261
  
"""

import sys
import datetime
assert len(sys.argv) > 1, "Must provide simulator as argument"
simulator = sys.argv[-1]
#import pyNN.nest2 as sim
exec("import pyNN.%s as sim" % simulator)
import pyNN.random as random
from numpy import exp, cos, pi, sin, arcsin
import numpy
import babble
babble.set_simulator(sim)


# =-=-= Global Parameters =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

seed             = 0           # Seed for the random number generator
ncells           = 30          # Number of input spike trains per layer
pconnect         = 1.0         # Connection probability
wmax             = 0.02        # Maximum synaptic weight
f_winhib         = 0.0         # Inhibitory weight = f_winhib*wmax (fixed)
f_wtr            = 1.0         # Max training weight = f_wtr*wmax
min_delay        = 0.1         # Non-zero minimum synaptic delay
syndelay         = 0           # Synaptic delay relative to min_delay
tauLTP           = 20          # (ms) Time constant for LTP
tauLTD           = 20          # (ms) Time constant for LTD
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
k                = [0.0]       # Function parameter(s)
wtr_square       = True        # Sets square or bell-shaped profile for T-->O weights
wtr_sigma        = 0.15        # Width parameter for Training-->Output weights
noise            = 1           # Noise parameter
histbins         = 100         # Number of bins for weight histograms
record_spikes    = True        # Whether or not to record spikes
record_v         = 0           # Number of output cells to record membrane potential from
wfromfile        = False       # if positive, read connections/weights from file
infile           = ""          # File to read connections/weights from
tstop            = 1e6         # (ms)
trw              = 1e5         # (ms) Time between reading input spikes/printing weights
numhist          = 10          # Number of histograms between each weight printout
label            = "bfstdp_"   # Extra label for labelling output files
datadir          = ""          # Sub-directory of Data for writing output files
tau_m            = 20          # Membrane time constant

# =-=-= Create utility objects  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

rng = random.NumpyRNG(seed)

# =-=-= Procedures =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Procedures to create the network --------------------------------------------

def build_network():
    global conn, cellLayers
    
    sim.Timer.start()
    sim.setup(min_delay=min_delay, debug=True)
    
    print "Creating network layers (time %g s)" % sim.Timer.elapsedTime()
    
    TRANSFORMATIONS = {
        None:   lambda x,a: x,
        "mul":  lambda x,a: a*x,
        "sin":  lambda x,a: 0.5*(sin(2*pi*x + a) + 1),
        "sq":   lambda x,a: x*x,
        "asin": lambda x,a: arcsin(2*x-1)/pi, 
        "sinn": lambda x,a: 0.5*(sin(2*pi*a*x) + 1),
    }
    
    cellParams = {}
    cellParams['Input'] = {
        'Rmax': Rmax, 'Rmin': Rmin, 'Rsigma': Rsigma, 'alpha': 1.0,
        'correlation_time': correlation_time, 'transform': None
    }
    cellParams['Training'] = {
        'Rmax': Rmax, 'Rmin': Rmin, 'Rsigma': Rsigma, 'alpha': alpha,
        'correlation_time': correlation_time,
        'transform': lambda x: TRANSFORMATIONS[funcstr](x, *k)
    }
    cellParams['Output'] = {
        'v_rest':     0,      'cm':        3.18,   'tau_m':     tau_m,
        'tau_refrac': 0.0,    'tau_syn_E': 5.0,    'tau_syn_I': 15.0,  
        'i_offset':   0.0,    'v_reset':   0.0,    'v_thresh':  1.0,
        'v_init':     0.0,
    }
    cellParams['Background'] = {
        'rate': bgRate, 'duration': tstop
    }

    # Create network layers
    cellLayers = {}
    for layer in 'Input','Training':
        cellLayers[layer] = babble.BabblingPopulation(ncells, **cellParams[layer])
    cellLayers['Output'] = sim.Population(ncells, sim.IF_curr_exp, cellParams['Output'])
    cellLayers['Background'] = sim.Population(ncells, sim.SpikeSourcePoisson, cellParams['Background'])
    
    # Create synaptic connections
    print "Creating synaptic connections (time %g s)" % sim.Timer.elapsedTime()
    
    # Turn on STDP for Input-->Output connections
    print "  Defining STDP configuration for Input-->Output connections"
    aLTD = B*aLTP*tauLTP/tauLTD
    stdp_model = sim.STDPMechanism(timing_dependence=sim.SpikePairRule(tau_plus=tauLTP, tau_minus=tauLTD),
                                   weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=wmax,
                                                                                  A_plus=aLTP,
                                                                                  A_minus=aLTD))
    synapse_dynamics = sim.SynapseDynamics(fast=None, slow=stdp_model)
    
    if wfromfile: # read connections from file
        connectors = { 'IO': sim.FromFileConnector("%s.connIO.conn" % infile),
                       'TO': sim.FromFileConnector("%s.connTO.conn" % infile),
                       'OO': f_winhib and sim.FromFileConnector("%s.connOO.conn" % infile) } # only defined if Inhibitory weight is non-zero
    else:         # or generate them according to the rules specified
        c = sim.FixedProbabilityConnector(pconnect)
        connectors = { 'IO': c,
                       'TO': c,
                       'OO': f_winhib and sim.AllToAllConnector(allow_self_connections=False) }
        
    conn = {}
    conn['IO'] = sim.Projection(cellLayers['Input'], cellLayers['Output'],
                                method=connectors['IO'], rng=rng,
                                synapse_dynamics=synapse_dynamics)
    conn['TO'] = sim.Projection(cellLayers['Training'], cellLayers['Output'],
                                method=connectors['TO'], rng=rng,
                                synapse_dynamics=None)
    if f_winhib != 0:
        conn['OO'] = sim.Projection(cellLayers['Output'], cellLayers['Output'],
                                    method=connectors['OO'],
                                    synapse_dynamics=None)
    conn['Bg'] = sim.Projection(cellLayers['Background'], cellLayers['Output'],
                                method=sim.OneToOneConnector(weights=bgWeight))

    if not wfromfile:
        initial_weight_distribution = random.RandomDistribution('uniform', (0.0,wmax), rng)
        conn['IO'].randomizeWeights(initial_weight_distribution)
        set_training_weights(conn['TO'])
        if syndelay < 0:
          conn['IO'].setDelays(min_delay + -1*syndelay)
          conn['TO'].setDelays(min_delay)
        elif syndelay > 0:
            conn['IO'].setDelays(min_delay)
            conn['TO'].setDelays(min_delay + syndelay)
        
        if f_winhib:
            conn['OO'].setWeights(wmax*f_winhib)

    # Turn on recording of spikes
    if record_spikes:
        for layer in ('Input','Output','Training'):
            cellLayers[layer].record()
    if record_v:
        cellLayers['Output'].record_v(int(record_v))

    print "Finished set-up (time %g s)" % sim.Timer.elapsedTime()

# Utility procedures ----------------------------------------------------------

def get_fileroot():
    global datadir, label, funcstr, simulator
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fileroot = "Data/%s/%s_%s_%s" % (datadir, label, simulator, funcstr)
    for param in k:
        fileroot += "-%3.1f" % param
    fileroot += "_%s" % timestamp
    return fileroot

# Procedures to set weights ---------------------------------------------------

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

# Procedures for writing results to file --------------------------------------

def save_parameters(fileroot):
    filename = "%s.param" % fileroot
    f = open(filename, 'w')
    lines = ["# Parameters for bfstdp.py"]
    lines += ["%-17s = %d" % (p, eval(p)) for p in ("seed", "ncells")]
    lines += ["%-17s = %f" % (p, eval(p)) for p in (
        "pconnect", "wmax", "f_winhib", "f_wtr", "syndelay", "tauLTP", "tauLTD",
        "B","aLTP", "Rmax", "Rmin", "Rsigma", "alpha", "correlation_time",
        "bgWeight", "bgRate", "wtr_sigma", "noise", "tau_m")]
    lines += ["%-17s = %s" % (p, eval(p)) for p in ("wtr_square", "k")]
    lines += ['%-17s = "%s"' % (p, eval(p)) for p in ("funcstr",)]
    if wfromfile:
        lines += ['%-17s = "%s"' % (p, eval(p)) for p in ("infile",)]
    f.write("\n".join(lines))
    f.close()

def print_v(fileroot):
    for layer in ('Output',):
        cellLayers[layer].print_v("%s.cell%s.v" % (fileroot,layer))

def print_rasters(fileroot):
    for layer in ('Input','Output','Training'):
        cellLayers[layer].printSpikes("%s.cell%s.ras" % (fileroot,layer))

def print_weights(fileroot, projection_label):
    conn[projection_label].printWeights("%s.conn%s.w" % (fileroot, projection_label),
                                        format='array')

def save_connections(fileroot):
     for projection_label in conn.keys():
       conn[projection_label].saveConnections("%s.conn%s.conn" % (fileroot, projection_label))

def print_weight_distribution(histfileobj):
    # Pointless to calculate distribution for inhibitory weights (i=1,2)
    hist, bins = conn['IO'].weightHistogram(min=0, max=wmax, nbins=histbins)
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
#      nspikes_post = cellLayers[2].cell[i].spiketimes.size()
#      if (nspikes_post > 0) {
#	for j = 0, nspikes_post-1 {
#	  tpost = cellLayers[2].cell[i].spiketimes.x[j]
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
    fileroot = get_fileroot()
    thist = int(trw/numhist)
    histfileobj = open("%s.connIO.whist" % fileroot, "w")
    save_parameters(fileroot)
    
    fileroot2 = "%s_0" % fileroot
    print_weights(fileroot2, 'IO')
    print_weights(fileroot2, 'TO')
    save_connections(fileroot2)
    
    i = 0; j = 0
    #setup_weight_plot()
    #plot_weights(conn[0])
    sim.Timer.reset()
    t = 0
    while t < tstop:
        fileroot2 = "%s_%d" % (fileroot, j*thist)
        print_weight_distribution(histfileobj)
        if i == numhist:
            print_weights(fileroot2, 'IO')
            i = 0
        print "--- Simulated %d seconds in %d seconds (%d%%)\r" % (int(t/1000), sim.Timer.elapsedTime(), int(100*t/tstop)),
        sys.stdout.flush()
        i += 1
        j += 1
        for layer in 'Input', 'Training':
            cellLayers[layer].generate_spikes(thist, sync=cellLayers['Input'])
        t = sim.run(thist)
        #plot_weights(conn[0])
    
    print "--- Simulated %d seconds in %d seconds\n" % (int(t/1000), sim.Timer.elapsedTime())
    
    fileroot2 = "%s_%d" % (fileroot, j*thist)
    print_weights(fileroot2, 'IO')
    print_weights(fileroot2, 'TO') # for debugging. Should not have changed since t = 0
    print_weight_distribution(histfileobj)
    save_connections(fileroot2)
    
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
    
    print_rasters(fileroot)
    if record_v:
        print_v(fileroot)
    
    histfileobj.close()
    print "Training complete. Time ", sim.Timer.elapsedTime()
    #calc_delta_t(1.0,1000,0)
  
    
# =-=-= Initialize the network =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    build_network()
    print "Running training ..."
    run_training()


