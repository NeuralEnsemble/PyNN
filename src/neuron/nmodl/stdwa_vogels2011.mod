COMMENT
Spike Timing Dependent Weight Adjuster implementing the rule from

  Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W (2011)
  Inhibitory plasticity balances excitation and inhibition in sensory pathways
  and memory networks. Science 334:1569-73

  http://dx.doi.org/10.1126/science.1211095
  
also see http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=143751


Andrew Davison, UNIC, CNRS, 2013
ENDCOMMENT

NEURON {
    POINT_PROCESS StdwaVogels2011
    RANGE interval, tlast_pre, tlast_post
    RANGE deltaw, wmax, wmin, tau, eta, rho, on
    RANGE allow_update_on_post
    POINTER wsyn
}

ASSIGNED {
    interval    (ms)    : since last spike of the other kind
    tlast_pre   (ms)    : time of last presynaptic spike
    tlast_post  (ms)    : time of last postsynaptic spike
    deltaw              : change in weight
    wsyn                : weight of the synapse
    alpha
}

INITIAL {
    interval = 0
    tlast_pre = -1e12
    tlast_post = -1e12
    deltaw = 0
}

PARAMETER {
    tau  = 20 (ms)           : decay time constant for exponential part of f
    wmax = 1                 : maximum value of synaptic weight
    wmin = 0                 : minimum value synaptic weight
    eta  = 1e-10             : learning rate
    rho  = 3e-3              : strength of non-Hebbian synaptic depression relative to Hebbian potentiation.
    on   = 1                 : allows learning to be turned on and off
    allow_update_on_post = 1 : if this is true, we update the weight on receiving both pre- and post-synaptic spikes
                             : if it is false, weight updates are accumulated and applied only for a pre-synaptic spike
}

NET_RECEIVE (w) {
    if (w >= 0) {                           : this is a pre-synaptic spike
        interval = tlast_post - t
        tlast_pre = t
        alpha = 2*rho*tau
        deltaw = deltaw + eta*(exp(interval/tau) - alpha)
    } else {                                : this is a post-synaptic spike
        interval = t - tlast_pre
        tlast_post = t
        deltaw = deltaw + eta*exp(-interval/tau)
    }
    if (on) {
        if (w >= 0 || allow_update_on_post) {
            wsyn = wsyn + deltaw
            if (wsyn > wmax) {
                wsyn = wmax
            }
            if (wsyn < wmin) {
                wsyn = wmin
            }
            deltaw = 0.0
        }
    }
}
