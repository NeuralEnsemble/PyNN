: Insert in a passive compartment to get an adaptive-exponential (Brette-Gerstner)
: integrate-and-fire neuron with a refractory period.
: This calculates the adaptive current, sets the membrane potential to the
: correct value at the start and end of the refractory period, and prevents spikes
: during the period by clamping the membrane potential to the reset voltage with
: a huge conductance.
:
: Reference:
:
: Brette R and Gerstner W. Adaptive exponential integrate-and-fire
:   model as an effective description of neuronal activity. 
:   J. Neurophysiol. 94: 3637-3642, 2005.
:  
: Implemented by Andrew Davison. UNIC, CNRS, March 2009.
: $Id:$

NEURON {
    POINT_PROCESS AdExpIF
    RANGE v_reset, t_refrac, v_spike, v_thresh, v_peak, spikewidth
    RANGE w, w_init
    RANGE a, b, tau_w, EL, GL, delta
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

PARAMETER {
    v_thresh = -50   (mV)   : spike threshold for exponential calculation purposes
    v_reset  = -60   (mV)   : reset potential after a spike
    v_spike  = -40   (mV)   : spike detection threshold
    v_peak   = 0     (mV)   : peak of spike
    t_refrac = 1     (ms)   : refractory period
    g_on    = 1e12   (uS)   : refractory clamp conductance
    spikewidth = 1e-12 (ms) : must be less than trefrac
    
    a 	    = 0.004  (uS)   : level of adaptation
    b	    = 0.0805 (nA)   : increment of adaptation
    tau_w   = 144    (ms)   : time constant of adaptation
    EL	    = -70.6  (mV)   : leak reversal (must be equal to e_pas)
    GL	    = 0.03   (uS)   : leak conductance (must be equal to g_pas(S/cm2)*membrane area(um2)*1e-2)
    delta   = 2      (mV)   : steepness of exponential approach to threshold

    w_init  = 0      (nA)
}


ASSIGNED {
    v (mV)
    i (nA)
    i_refrac (nA)
    i_exp (nA)
    g_refrac (uS)
    refractory
}

STATE {
    w  (nA)
}

INITIAL {
    g_refrac = 0
    net_send(0,4)
    w = w_init
}

BREAKPOINT {
    SOLVE states METHOD cnexp   :derivimplicit
    i_refrac = g_refrac*(v-v_reset)
    i_exp = - GL*delta*exp((v-v_thresh)/delta)
    i = i_exp + w + i_refrac
    :printf("BP: t = %f  dt = %f  v = %f  vv = %f w = %f  i_refrac = %f  i_exp = %f  i = %f  delta_v = %f\n", t, dt, v, vv, w, i_refrac, i_exp, i, delta_v)
}


DERIVATIVE states {		: solve eq for adaptation variable
    w' = (a*(v-EL) - w)/tau_w
}

NET_RECEIVE (weight) {
    if (flag == 1) {        : beginning of spike
        v = v_peak
        w = w + b
        net_send(spikewidth, 2)
        net_event(t)
        printf("spike: t = %f  v = %f   w = %f   i = %f\n", t, v, w, i)
    } else if (flag == 2) { : end of spike, beginning of refractory period
        v = v_reset
        g_refrac = g_on
        if (t_refrac > spikewidth) {
            net_send(t_refrac-spikewidth, 3)
        } else { : also the end of the refractory period
            g_refrac = 0
        }
        printf("refrac: t = %f  v = %f   w = %f   i = %f\n", t, v, w, i)
    } else if (flag == 3) { : end of refractory period
        v = v_reset
        g_refrac = 0
        printf("end_refrac: t = %f  v = %f   w = %f   i = %f\n", t, v, w, i)
    } else if (flag == 4) { : watch membrane potential
        WATCH (v > v_spike) 1
    }
}