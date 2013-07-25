: Insert in a passive compartment to get an integrate-and-fire neuron
: with a refractory period.
: Note that this only sets the membrane potential to the correct value
: at the start and end of the refractory period, and prevents spikes
: during the period by clamping the membrane potential to the reset
: voltage with a huge conductance.
:
: Andrew P. Davison. UNIC, CNRS, May 2006.

NEURON {
    POINT_PROCESS ResetRefrac
    RANGE vreset, trefrac, vspike, vthresh
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

PARAMETER {
    vthresh = -50   (mV)    : spike threshold
    vreset  = -60   (mV)    : reset potential after a spike
    vspike  = 40    (mV)    : spike height (mainly for graphical purposes)
    trefrac = 1     (ms)
    g_on    = 1e12  (uS)
    spikewidth = 1e-12 (ms) : must be less than trefrac. Check for this?
}


ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    refractory
}

INITIAL {
    g = 0
    net_send(0,4)
}

BREAKPOINT {
    i = g*(v-vreset)
}

NET_RECEIVE (weight) {
    if (flag == 1) {        : beginning of spike
        g = g_on
        state_discontinuity(v,vspike)
        net_send(spikewidth,2)
        net_event(t)
    } else if (flag == 2) { : end of spike, beginning of refractory period
        state_discontinuity(v,vreset)
        if (trefrac > spikewidth) {
            net_send(trefrac-spikewidth,3)
        } else { : also the end of the refractory period
            g = 0
        }
    } else if (flag == 3) { : end of refractory period
        state_discontinuity(v,vreset)
        g = 0
    } else if (flag == 4) { : watch membrane potential
         WATCH (v > vthresh) 1
    }
}