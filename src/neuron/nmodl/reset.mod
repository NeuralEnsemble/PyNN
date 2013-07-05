: Insert in a passive compartment to get an integrate-and-fire neuron
: (no refractory period).
: Andrew P. Davison. UNIC, CNRS, May 2006.

NEURON {	
	POINT_PROCESS Reset
	RANGE vreset, vspike
}

UNITS {
	(mV) = (millivolt)
}	

PARAMETER {
	vreset	= -60	(mV)	: reset potential after a spike
	vspike  = 40    (mV)    : spike height (mainly for graphical purposes)
}

ASSIGNED {
	v (millivolt)
}

NET_RECEIVE (weight) {
	if (flag == 1) {
		v = vreset
	} else {
		v = vspike
		net_send(1e-12,1)  : using variable time step, this should allow the spike to be detected using threshold crossing
		net_event(t)
	}
}
