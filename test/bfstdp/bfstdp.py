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

Andrew P. Davison, UNIC, CNRS, July 2004-May 2006
"""

import os
import pyNN.neuron as sim
import pyNN.random as random

sim.Timer.start()

#cvode = new CVode() - in setup()
#xopen("plotweights.hoc")

# =-=-= Global Parameters =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

seed             = 0           # Seed for the random number generator
ncells           = 30          # Number of input spike trains per layer
pconnect         = 1.0         # Connection probability
wmax             = 0.02        # Maximum synaptic weight
f_winhib         = 0.0         # Inhibitory weight = f_winhib*wmax (fixed)
f_wtr            = 1.0         # Max training weight = f_wtr*wmax
syndelay         = 0.1         # Synaptic delay
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
wtr_square       = 1           # Sets square or bell-shaped profile for T-->O weights
wtr_sigma        = 0.15        # Width parameter for Training-->Output weights
noise            = 1           # Noise parameter
histbins         = 100         # Number of bins for weight histograms
record_spikes    = False       # Whether or not to record spikes
wfromfile        = False       # if positive, read connections/weights from file
infile           = ""          # File to read connections/weights from
tstop            = 1e7         # (ms)
trw              = 1e5         # (ms) Time between reading input spikes/printing weights
numhist          = 10          # Number of histograms between each weight printout
label            = "bfstdp_demo_" # Extra label for labelling output files
datadir          = ""          # Sub-directory of Data for writing output files
tau_m            = 20          # Membrane time constant

# =-=-= Create utility objects  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

rng = random.NumpyRNG(seed)

# =-=-= Create the network  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

sim.setup(min_delay=abs(syndelay), use_cvode=True)
sim.hoc_execute('nrn_load_dll("%s/i686/.libs/libnrnmech.so")' % os.getcwd())

# Input spike trains are implemented using NetStimVR2s.
print "Creating network layers (time %g s)" % sim.Timer.elapsedTime()

cellParams = {
    'tau_m': tau_m,
    'tau_e': 5,
    'tau_i1': 10,
    'tau_i2': 15
}

# Create network layers
for layer in 0,1:
    cellLayer[layer] = sim.Population(ncells, "NetStimVR2")
    cellLayer[layer].set("noise", 1)
    for i in range(ncells):
        cellLayer[layer].cell[i].theta = i/ncells
    
sim.hoc_execute("Rmax_NetStimVR2 = %g" % Rmax)
sim.hoc_execute("Rmin_NetStimVR2 = %g" % Rmin)
sim.hoc_execute("sigma_NetStimVR2 = %g" % Rsigma)

cellLayer[0].set("transform",0)
cellLayer[0].set("prmtr",0)

TRANSFORMATIONS = {
    "":     0,
    "mul":  1,
    "sin":  2,
    "sq":   3,
    "asin": 4, 
    "sinn": 5
}
cellLayer[1].set("transform", TRANSFORMATIONS[funcstr])
cellLayer[1].set("prmtr", k[0])
cellLayer[1].set("alpha", alpha)

# due to the setpointer, need to encapsulate this in a hoc template, I think
#spikecontrol = new ControlNSVR2(0.5)
#spikecontrol.tau_corr = correlation_time
#spikecontrol.seed(seed)
#setpointer spikecontrol.thetastim, thetastim_NetStimVR2
#setpointer spikecontrol.tchange, tchange_NetStimVR2

cellLayer[2] = Population(ncells, "IntFire4nc", cellParams)

# Create synaptic connections
print "Creating synaptic connections (time %g s)" % sim.Timer.elapsedTime()

initial_weight_distribution = random.RandomDistribution('uniform', (0,wmax), rng)

# Turn on STDP for Input-->Output connections

print "  Defining STDP configuration for Input-->Output connections"
aLTD = B*aLTP*tauLTP_StdwaSA/tauLTD_StdwaSA
stdp_model = STDPMechanism(timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0),
                           weight_dependence=AdditiveWeightDependence(w_min=0, w_max=wmax,
                                                                      A_plus=aLTP,
                                                                      A_minus=aLTD))

if wfromfile: # read connections from file
    conn[0] = Projection(cellLayer[0], cellLayer[2],
                         method=sim.FromFileConnector("%s.conn1.conn" % infile),
                         source="",
                         target="syn",
                         synapse_dynamics=stdp_model)
    conn[1] = Projection(cellLayer[1], cellLayer[2],
                         method=sim.FromFileConnector("%s.conn2.conn" % infile),
                         source="",
                         target="syn",
                         synapse_dynamics=None)
    if f_winhib != 0:
      filename = "%s.conn2.conn" % infile
      conn[2] = Projection(cellLayer[2], cellLayer[2],
                           method=sim.FromFileConnector(filename),
                           source="syn",
                           target="syn")
else:         # or generate them according to the rules specified
    connector = sim.FixedProbabilityConnector(p_connect) # can you reuse a connector?
    conn[0] = Projection(cellLayer[0], cellLayer[2],
                         method=sim.FixedProbabilityConnector(p_connect),
                         source="syn",
                         target="syn",
                         rng=rng,
                         synapse_dynamics=stdp_model)
    conn[0].randomizeWeights(initial_weight_distribution)
    conn[1] = Projection(cellLayer[1], cellLayer[2],
                         method=sim.FixedProbabilityConnector(p_connect),
                         source="syn",
                         target="syn",
                         rng=rng,
                         synapse_dynamics=None)
    
    if syndelay < 0:
      conn[0].setDelays(-1*syndelay)
      conn[1].setDelays(0)
    elif syndelay > 0:
        conn[0].setDelays(0)
        conn[1].setDelays(syndelay)

    if f_winhib != 0:
        conn[2] = Population(cellLayer[2], cellLayer[2],
                             method=sim.AllToAllConnector(allow_self_connections=False),
                             source="syn",
                             target="syn")
        conn[2].setWeights(wmax*f_winhib)



# Set background input
print "Setting background activity (time ", stopsw(), "s)"
sprint(command,"%f, %f, 0, 1, 1e12",bgWeight,bgRate)
cellLayer[2].call("set_background",command)

# Turn on recording of spikes
if (record_spikes) {
  cellLayer[2].call("record","1")
  for i = 0,ncells-1 {
    conn[0].nc[i][i].record(spikerec[0].x[i])
    conn[1].nc[i][i].record(spikerec[1].x[i])
  }
}


# =-=-= Procedures =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Utility procedures ----------------------------------------------------------

proc set_fileroot() { local i
  system("date '+%Y%m%d_%H%M' > starttime")
  fileobj[0].ropen("starttime")
  fileobj[0].scanstr(save_fileroot)
  fileobj[0].close()
  sprint(fileroot,"Data/%s/%s%s",datadir,label,funcstr)
  for i = 0, nfuncparam-1 {
    sprint(fileroot,"%s-%3.1f",fileroot,k[i])
  }
  sprint(fileroot,"%s_%s",fileroot,save_fileroot)
  print "fileroot = ", fileroot
}

# Procedures to read input spike trains from file -----------------------------

# Procedures to set weights ---------------------------------------------------

proc set_training_weights() { local i, j, d
  # Set the Training-->Output weights
  
  for i = 0, ncells-1 {
    for j = 0, ncells-1 {
      if(object_id(conn[1].nc[i][j])) {
	d = i-j
	if (d > ncells/2)  { d = ncells - d }
	if (d < -ncells/2) { d = ncells + d }
	if (wtr_square) {
	  if (d <= wtr_sigma*ncells && d >= -wtr_sigma*ncells) {
	    conn[1].nc[i][j].weight = f_wtr*wmax
	  }
	} else {
	  conn[1].nc[i][j].weight = f_wtr*wmax*exp( (cos(2*PI*d/ncells) - 1) / (wtr_sigma*wtr_sigma) )
	}
      }
    }
  }
}

# Procedures for writing results to file --------------------------------------

proc save_parameters() { local i
  sprint(filename,"%s.param",fileroot)
  fileobj[0].wopen(filename)
  fileobj[0].printf("# Parameters for bfstdp_nsvr2.hoc\n")
  fileobj[0].printf("%-17s = %d\n","seed",seed)
  fileobj[0].printf("%-17s = %d\n","ncells",ncells)
  fileobj[0].printf("%-17s = %f\n","pconnect",pconnect)
  fileobj[0].printf("%-17s = %f\n","wmax",wmax)
  fileobj[0].printf("%-17s = %f\n","f_winhib",f_winhib)
  fileobj[0].printf("%-17s = %f\n","f_wtr",f_wtr)
  fileobj[0].printf("%-17s = %f\n","syndelay",syndelay)
  fileobj[0].printf("%-17s = %f\n","tauLTP_StdwaSA",tauLTP_StdwaSA)
  fileobj[0].printf("%-17s = %f\n","tauLTD_StdwaSA",tauLTD_StdwaSA)
  fileobj[0].printf("%-17s = %f\n","B",B)
  fileobj[0].printf("%-17s = %f\n","aLTP",aLTP)  
  fileobj[0].printf("%-17s = %f\n","Rmax",Rmax)
  fileobj[0].printf("%-17s = %f\n","Rmin",Rmin)
  fileobj[0].printf("%-17s = %f\n","Rsigma",Rsigma)
  fileobj[0].printf("%-17s = %f\n","alpha",alpha)
  fileobj[0].printf("%-17s = %f\n","correlation_time",correlation_time)
  fileobj[0].printf("%-17s = %f\n","bgWeight",bgWeight)
  fileobj[0].printf("%-17s = %f\n","bgRate",bgRate)
  fileobj[0].printf("%-17s = \"%s\"\n","funcstr",funcstr)
  fileobj[0].printf("%-17s = %f\n","nfuncparam",nfuncparam)
  for i = 0, nfuncparam-1 {
    fileobj[0].printf("%-14s[%d] = %f\n","k",i,k[i])
  }
  fileobj[0].printf("%-17s = %f\n","wtr_square",wtr_square)
  fileobj[0].printf("%-17s = %f\n","wtr_sigma",wtr_sigma)
  fileobj[0].printf("%-17s = %f\n","noise",noise)
  fileobj[0].printf("%-17s = %f\n","tau_m",tau_m)
  if (wfromfile) {
    fileobj[0].printf("%-17s = \"%s\"\n","infile",infile)
  }
  fileobj[0].close()
}

proc print_rasters() { local i,j,k
  # Write spike times to files.
  # Plot using 
  #   gnuplot> plot "<fileroot>.input1.ras" u 1:2 w d
  
  if (record_spikes) {
    for i = 0,1 {
      sprint(filename,"%s.cell%d.ras",fileroot,i+1)
      $o1.wopen(filename)
      for j = 0,ncells-1 {
	for k = 0,spikerec[i].x[j].size()-1 {
	  $o1.printf("%15.5g\t%d\n",spikerec[i].x[j].x[k],j)
	}
	$o1.printf("\n")
      }
      $o1.close()
    }
    sprint(filename,"%s.cell3.ras",fileroot)
    $o1.wopen(filename)
    cellLayer[2].print_spikes($o1)
    $o1.close()
  }
}

proc print_weights() { local i
  sprint(filename,"%s.conn%d.w",fileroot,$1+1)
  fileobj[0].wopen(filename)
  conn[$1].print_weights(fileobj[0])
  fileobj[0].close()
}

proc save_connections() { local i
  for i = 0,2-(f_winhib==0) {
    sprint(filename,"%s.conn%d.conn",fileroot,i+1)
    fileobj[0].wopen(filename)
    conn[i].save_connections(fileobj[0])
    fileobj[0].close()
  }
}

proc print_weight_distribution() { local i
  # Pointless to calculate distribution for inhibitory weights (i=1,2)
  conn[0].print_weight_hist(histfileobj,histbins,1)
}

proc print_delta_t() { local i,ii, histbins, range, total_size
  binwidth = $1 # ms
  range = $2
  histbins = 2*range+1
  deltat_hist = new Vector(histbins)
  for layer = 0,1 {
    total_size = deltat_vec[layer][0].size() + deltat_vec[layer][1].size() + deltat_vec[layer][2].size()
    for ii = 0,2 {
      deltat_hist.hist(deltat_vec[layer][ii],-range-0.5,histbins,binwidth)
      if ($3 == 1) deltat_hist.div(total_size)
      sprint(filename,"%s.conn%d.deltat%d",fileroot,layer+1,ii)
      fileobj.wopen(filename)
      for i = 0, histbins-1 { #print in a column
	fileobj.printf("%g\t%g\n",-range+binwidth*i,deltat_hist.x[i])
      }
      #deltat_vec.printf(fileobj)
      fileobj.close()
    }
  }
}

# Procedures that process recorded data ---------------------------------------

proc calc_delta_t() { local i,j,k,l,ii, nspikes_post, nspikes_pre, deltat, d
  # Calculate the distribution of spike-time differences (post-pre)
  # in three classes: connections for which d < 0.1, d < 0.2, d >= 0.2
  if (record_spikes) {
    for ii = 0,2 {
      for layer = 0,1 {
	deltat_vec[layer][ii] = new Vector(1e6)
	m[layer][ii] = 0
      }
    }
    for i = 0,ncells-1 {
      nspikes_post = cellLayer[2].cell[i].spiketimes.size()
      if (nspikes_post > 0) {
	for j = 0, nspikes_post-1 {
	  tpost = cellLayer[2].cell[i].spiketimes.x[j]
	  for k = 0,ncells-1 {
	    for layer = 0,1 {
	      if (layer==0) {
		d  = i/ncells - (sin(2*PI*k/ncells)+1)/2
	      } else {
		d = i/ncells - k/ncells
	      }
	      if (d < -0.5) d += 1
	      if (d >= 0.5) d -= 1
	      d = abs(d)
	      if (d < 0.1) {
		ii = 0
	      } else {
		if (d < 0.2) {
		  ii = 1
		} else {
		  ii = 2
		}
	      }
	      nspikes_pre = spikerec[layer].x[k].size()
	      if (nspikes_pre > 0) {
		for l = 0, nspikes_pre-1 {
		  deltat = tpost - spikerec[layer].x[k].x[l]
		  if (deltat < $2 && deltat > -1*$2) {
		    deltat_vec[layer][ii].x[m[layer][ii]] = deltat
		    m[layer][ii] += 1
		    if (m[layer][ii] >= deltat_vec[layer][ii].size()-1) {
		      deltat_vec[layer][ii].resize(2*deltat_vec[layer][ii].size)
		      printf("deltat_vec[%d][%d] resized\n",layer,ii)
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    printf("Spike pairs: %d,%d  %d,%d  %d,%d\n",m[0][0],m[1][0],m[0][1],m[1][1],m[0][2],m[1][2])
    for ii = 0,2 {
      deltat_vec[0][ii].resize(m[0][ii])
      deltat_vec[1][ii].resize(m[1][ii])
    }
    print_delta_t($1,$2,$3)
    
  }
}



# Procedures that run simulations ---------------------------------------------

proc run_training() { local i, j, fileopen, thist
  # Training the network. The weight histogram is written to
  # file every trw ms. The weights are written to file every
  # thist = trw/numhist ms. The spike-times of the network
  # cells are written to file at the end.
  

  on_StdwaSA = 1
  thist = int(trw/numhist)

  sprint(filename,"%s.conn1.whist",fileroot)
  histfileobj.wopen(filename)
  
  save_parameters()
  save_fileroot = fileroot
  sprint(fileroot,"%s_%d",save_fileroot,0)
  print_weights(0)
  print_weights(1)
  save_connections()
  
  i = 0
  j = 0

  running_ = 1
  stoprun = 0
  setup_weight_plot()
  finitialize(-65)
  plot_weights(conn[0])
  starttime = startsw()
  while (t < tstop && stoprun == 0) {
    sprint(fileroot,"%s_%d",save_fileroot,j*thist)
    print_weight_distribution()
    if (i == numhist) {
      print_weights(0)
      i = 0
      printf("--- Simulated %d seconds in %d seconds\r",int(t/1000),startsw()-starttime)
      flushf()
    }
    i += 1
    j += 1
    while (t < j*thist) {
      fadvance()
    }
    #continuerun(j*thist)
    plot_weights(conn[0])
  }
  printf("--- Simulated %d seconds in %d seconds\n",int(t/1000),stopsw())
  
  sprint(fileroot,"%s_%d",save_fileroot,j*thist)
  print_weights(0)
  print_weights(1) # for debugging. Should not have changed since t = 0
  print_weight_distribution()
  save_connections()
  
  fileroot = save_fileroot
  
  # This corrects the pre-synaptic spiketimes for syndelay.
  # This is necessary because nc.record records spike times at the source
  # whereas we want to know them at the target.
  
  if (syndelay < 0) {
    for i = 0,ncells-1 {
      spikerec[0].x[i].add(-1*syndelay)
    } 
  } else if (syndelay > 0) {
    for i = 0,ncells-1 {
      spikerec[1].x[i].add(syndelay)
    }
  }

  
  print_rasters(fileobj[0])
  
  histfileobj.close()
  print "Training complete. Time ", stopsw()
  calc_delta_t(1.0,1000,0)
}

# =-=-= Initialize the network =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

set_fileroot()
cvode.active(1)
cvode.use_local_dt(1)         # The variable time step method must be used.
cvode.condition_order(2)      # Improves threshold-detection.
set_training_weights()
#steps_per_ms = 10
#dt = 0.1

print "Finished set-up (time ", stopsw(), "s)"

print "Running training ..."

run_training()

