: Insert in a passive compartment to get an adaptive-exponential (Brette-Gerstner)
: integrate-and-fire neuron with a refractory period.
: This calculates the adaptive current, sets the membrane potential to the
: correct value at the start and end of the refractory period, and prevents spikes
: during the refractory period by clamping the membrane potential to the reset
: voltage with a huge conductance.
:
: Reference:
:
: Brette R and Gerstner W. Adaptive exponential integrate-and-fire
:   model as an effective description of neuronal activity. 
:   J. Neurophysiol. 94: 3637-3642, 2005.
:  
: Implemented by Andrew Davison. UNIC, CNRS, March 2009.

NEURON {
    POINT_PROCESS AdExpIF
    RANGE vreset, trefrac, vspike, vthresh, vpeak, spikewidth
    RANGE w, winit
    RANGE a, b, tauw, EL, GL, delta
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

PARAMETER {
    vthresh = -50   (mV)   : spike threshold for exponential calculation purposes
    vreset  = -60   (mV)   : reset potential after a spike
    vspike  = -40   (mV)   : spike detection threshold
    vpeak   = 0     (mV)   : peak of spike
    trefrac = 1     (ms)   : refractory period
    gon    = 1e6   (uS)   : refractory clamp conductance
    spikewidth = 1e-12 (ms) : must be less than trefrac
    
    a 	    = 0.004  (uS)   : level of adaptation
    b	    = 0.0805 (nA)   : increment of adaptation
    tauw   = 144    (ms)   : time constant of adaptation
    EL	    = -70.6  (mV)   : leak reversal (must be equal to e_pas)
    GL	    = 0.03   (uS)   : leak conductance (must be equal to g_pas(S/cm2)*membrane area(um2)*1e-2)
    delta   = 2      (mV)   : steepness of exponential approach to threshold

    winit  = 0      (nA)
}


ASSIGNED {
    v (mV)
    i (nA)
    irefrac (nA)
    iexp (nA)
    grefrac (uS)
    refractory
    spike_threshold (mV)
}

STATE {
    w  (nA)
}

INITIAL {
    grefrac = 0
    net_send(0,4)
    w = winit
    if (delta == 0) {
        spike_threshold = vthresh
    } else {
        spike_threshold = vspike
    }
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit  : cnexp
    irefrac = grefrac*(v-vreset)
    iexp = exp_current(v)
    i = iexp + w + irefrac
    :printf("BP: t = %f  dt = %f  v = %f  w = %f  irefrac = %f  iexp = %f  i = %f\n", t, dt, v, w, irefrac, iexp, i)
}

DERIVATIVE states {		: solve eq for adaptation variable
    w' = (a*(v-EL) - w)/tauw
}

FUNCTION exp_current(v) {  : handle the case where delta is 0 or very small
    if (delta == 0) {
        exp_current = 0
    } else if ((v - vthresh)/delta > 100) {
        exp_current = -exp(99)
    } else {
        exp_current = -GL*delta*exp((v-vthresh)/delta)
    }
}

NET_RECEIVE (weight) {
    if (flag == 1) {        : beginning of spike
        v = vpeak
        w = w + b
        net_send(spikewidth, 2)
        net_event(t)
        :printf("spike: t = %f  v = %f   w = %f   i = %f\n", t, v, w, i)
    } else if (flag == 2) { : end of spike, beginning of refractory period
        v = vreset
        grefrac = gon
        if (trefrac > spikewidth) {
            net_send(trefrac-spikewidth, 3)
        } else { : also the end of the refractory period
            grefrac = 0
        }
        :printf("refrac: t = %f  v = %f   w = %f   i = %f\n", t, v, w, i)
    } else if (flag == 3) { : end of refractory period
        v = vreset
        grefrac = 0
        :printf("end_refrac: t = %f  v = %f   w = %f   i = %f\n", t, v, w, i)
    } else if (flag == 4) { : watch membrane potential
        WATCH (v > spike_threshold) 1
    }
}