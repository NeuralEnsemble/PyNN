: Insert in a passive compartment to get an integrate-and-fire neuron
: with a refractory period.
: Note that this only sets the membrane potential to the correct value
: at the start and end of the refractory period, and prevents spikes
: during the period by clamping the membrane potential to the reset
: voltage with a huge conductance.
:
: Andrew P. Davison. UNIC, CNRS, May 2006.
: $Id: refrac.mod 14 2007-01-30 13:09:03Z apdavison $

NEURON {	
	POINT_PROCESS ResetRefrac
	RANGE vreset, trefrac, vspike
	NONSPECIFIC_CURRENT i
}

UNITS {
	(mV) = (millivolt)
	(nA) = (nanoamp)
	(umho) = (micromho)
}	

PARAMETER {
	vreset	= -60	(mV)	: reset potential after a spike
	vspike  = 40    (mV)    : spike height (mainly for graphical purposes)
	trefrac = 1     (ms)
	g_on    = 1e12  (umho)
	spikewidth = 1e-12 (ms)  : must be less than trefrac. Check for this?
}


ASSIGNED {
	v (mV)
	i (nA)
	g (umho)
	refractory
}

INITIAL {
	g = 0
	refractory = 0
}

BREAKPOINT {
	i = g*(v-vreset)
}	

NET_RECEIVE (weight) {
	if (flag == 1) { : end of spike, beginning of refractory period
		state_discontinuity(v,vreset)
		if (trefrac > spikewidth) {
			net_send(trefrac-spikewidth,2)
		} else { : also the end of the refractory period
			refractory = 0
			g = 0
		}
	} else {
		if (flag == 2) { : end of refractory period
			state_discontinuity(v,vreset)
			refractory = 0
			g = 0
		} else {
			if (!refractory) { : beginning of spike
				g = g_on
				state_discontinuity(v,vspike)
				refractory = 1
				net_send(spikewidth,1)
				net_event(t)
			}
		}
	}
}