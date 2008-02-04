COMMENT
Spike Timing Dependent Weight Adjuster
based on Song and Abbott, 2001, but with soft weight limits
Andrew Davison, UNIC, CNRS, 2003-2005
ENDCOMMENT

NEURON {
	POINT_PROCESS StdwaSoft
	RANGE interval, tlast_pre, tlast_post, M, P
	RANGE deltaw, wmax, wmin, aLTP, aLTD, wprune, tauLTP, tauLTD, on
	POINTER wsyn
}

ASSIGNED {
	interval	(ms)	: since last spike of the other kind
	tlast_pre	(ms)	: time of last presynaptic spike
	tlast_post	(ms)	: time of last postsynaptic spike
	M			: LTD function
	P			: LTP function
	deltaw			: change in weight
	wsyn			: weight of the synapse
}

INITIAL {
	interval = 0
	tlast_pre = 0
	tlast_post = 0
	M = 0
	P = 0
	deltaw = 0
}

PARAMETER {
	tauLTP  = 20	(ms)    : decay time for LTP part ( values from           )
	tauLTD  = 20	(ms)    : decay time for LTD part ( Song and Abbott, 2001 )
	wmax    = 1		: min and max values of synaptic weight
        wmin    = 0
	aLTP    = 0.001		: amplitude of LTP steps
	aLTD    = 0.00106	: amplitude of LTD steps
	on	= 1		: allows learning to be turned on and off globally
	wprune  = 0             : default is no pruning
}

NET_RECEIVE (w) {
	if (w >= 0) {				: this is a pre-synaptic spike
		P = P*exp((tlast_pre-t)/tauLTP) + aLTP
		interval = tlast_post - t	: interval is negative
		tlast_pre = t
		deltaw = (wsyn-wmin) * M * exp(interval/tauLTD)
	} else {				: this is a post-synaptic spike
		M = M*exp((tlast_post-t)/tauLTD) - aLTD
		interval = t - tlast_pre	: interval is positive
		tlast_post = t
		deltaw = (wmax-wsyn) * P * exp(-interval/tauLTP)
	}
	if (on) {
		if (wsyn > wprune) {
		  wsyn = wsyn + deltaw
		} else {
		  wsyn = 0
		}
	}
}
