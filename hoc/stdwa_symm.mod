COMMENT
Spike Timing Dependent Weight Adjuster
with symmetric functions (i.e. only depends on the absolute value of the
time difference, not on its sign.
Andrew Davison, UNIC, CNRS, September 2004
ENDCOMMENT

NEURON {
	POINT_PROCESS StdwaSymm
	RANGE interval, tlast_pre, tlast_post
	RANGE deltaw, wmax, f, tau_a, tau_b, a, on
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
	on	= 1		: allows learning to be turned on and off globally
}

NET_RECEIVE (w) {
	tas = tau_a * tau_a : do it here in case tau_a has been changed since the last spike

	if (w >= 0) {				: this is a pre-synaptic spike
		interval = tlast_post - t
		tlast_pre = t
		f = (1 - interval*interval/tas) * exp(interval/tau_b)
		deltaw = wmax * a * f
	} else {				: this is a post-synaptic spike
		interval = t - tlast_pre
		tlast_post = t
		f = (1 - interval*interval/tas) * exp(-interval/tau_b)
		deltaw = wmax * a* f
	}
	if (on) {
		wsyn = wsyn + deltaw
		if (wsyn > wmax) {
			wsyn = wmax
		}
		if (wsyn < 0) {
			wsyn = 0
		}
	}
}
