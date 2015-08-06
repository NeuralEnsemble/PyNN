COMMENT
Spike Timing Dependent Weight Adjuster
with symmetric functions (i.e. only depends on the absolute value of the
time difference, not on its sign.
Andrew Davison, UNIC, CNRS, 2004, 2009
ENDCOMMENT

NEURON {
	POINT_PROCESS StdwaSymm
	RANGE interval, tlast_pre, tlast_post
	RANGE deltaw, wmax, f, tau_a, tau_b, a, on
        RANGE allow_update_on_post
	POINTER wsyn
}

ASSIGNED {
	interval	(ms)	: since last spike of the other kind
	tlast_pre	(ms)	: time of last presynaptic spike
	tlast_post	(ms)	: time of last postsynaptic spike
	f                       : weight change function
	deltaw			: change in weight
	wsyn			: weight of the synapse
	tas             (ms2)   : tau_a squared
}

INITIAL {
	interval = 0
	tlast_pre = 0
	tlast_post = 0
	f = 0
	deltaw = 0
}

PARAMETER {
	tau_a   = 20 (ms)       : crossing point from LTP to LTD
	tau_b   = 15 (ms) 	: decay time constant for exponential part of f
	wmax    = 1		: min and max values of synaptic weight
	a       = 0.001		: step amplitude
	on	= 1		: allows learning to be turned on and off
        allow_update_on_post = 1 : if this is true, we update the weight on receiving both pre- and post-synaptic spikes
                                 : if it is false, weight updates are accumulated and applied only for a pre-synaptic spike
}

NET_RECEIVE (w) {
	tas = tau_a * tau_a : do it here in case tau_a has been changed since the last spike

	if (w >= 0) {				: this is a pre-synaptic spike
		interval = tlast_post - t
		tlast_pre = t
		f = (1 - interval*interval/tas) * exp(interval/tau_b)
		deltaw = deltaw + wmax * a * f
	} else {				: this is a post-synaptic spike
		interval = t - tlast_pre
		tlast_post = t
		f = (1 - interval*interval/tas) * exp(-interval/tau_b)
		deltaw = deltaw + wmax * a* f
	}
	if (on) {
            if (w >= 0 || allow_update_on_post) {
		wsyn = wsyn + deltaw
		if (wsyn > wmax) {
			wsyn = wmax
		}
		if (wsyn < 0) {
			wsyn = 0
		}
                deltaw = 0.0
            }
	}
}
