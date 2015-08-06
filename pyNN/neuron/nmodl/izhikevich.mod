: Izhikevich artificial neuron model from 
: EM Izhikevich "Simple Model of Spiking Neurons"
: IEEE Transactions On Neural Networks, Vol. 14, No. 6, November 2003 pp 1569-1572
:
: This NMODL file may not work properly if used outside of PyNN.
: Ted Carnevale has written a more complete, general-purpose implementation - see http://www.neuron.yale.edu/ftp/ted/devel/izhdistrib.zip

NEURON {
    POINT_PROCESS Izhikevich
    RANGE a, b, c, d, u, uinit, vthresh
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (nF) = (nanofarad)
}

INITIAL {
    u = uinit
    net_send(0, 1)
}

PARAMETER {
    a       = 0.02 (/ms)
    b       = 0.2  (/ms)
    c       = -65  (mV)   : reset potential after a spike
    d       = 2    (mV/ms)
    vthresh = 30   (mV)   : spike threshold
    Cm      = 0.001  (nF)
    uinit   = -14  (mV/ms)
}

ASSIGNED {
    v (mV)
    i (nA)
}

STATE { 
    u (mV/ms)
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit
    i = -Cm * (0.04*v*v + 5*v + 140 - u)
    :printf("t=%f, v=%f u=%f, i=%f, dv=%f, du=%f\n", t, v, u, i, 0.04*v*v + 5*v + 140 - u, a*(b*v-u))
}

DERIVATIVE states {
    u' = a*(b*v - u) 
}

NET_RECEIVE (weight (mV)) {
    if (flag == 1) {
        WATCH (v > vthresh) 2
    } else if (flag == 2) {
        net_event(t)
        v = c
        u = u + d
    } else { : synaptic activation
        v = v + weight
    }
}
