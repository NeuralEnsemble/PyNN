:    conductance based spike-frequency adaptation, and a conductance-based relative refractory
:    mechanism ... to be inserted in a integrate-and-fire neuron
:
:    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond
:    mean-adaptation and renewal theories. Neural Computation 19: 2958-3010.
:
:  
: Implemented from adexp.mod by Eilif Muller. EPFL-BMI, Jan 2011.

NEURON {
    POINT_PROCESS GsfaGrr
    RANGE vthresh
    RANGE q_r, q_s
    RANGE E_s, E_r, tau_s, tau_r
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
    (nS) = (nanosiemens)
}

PARAMETER {
    vthresh = -57   (mV)   : spike threshold
    q_r    = 3214.0   (nS)   : relative refractory quantal conductance
    q_s    = 14.48 (nS)   : SFA quantal conductance
    tau_s   = 110.0    (ms)   : time constant of SFA
    tau_r   = 1.97   (ms)   : time constant of relative refractory mechanism
    E_s     = -70    (mV)   : SFA reversal potential
    E_r     = -70    (mV)   : relative refractory period reversal potential
}


ASSIGNED {
    v (mV)
    i (nA)
}

STATE {
    g_s  (nS)
    g_r  (nS)
}

INITIAL {
    g_s = 0
    g_r = 0
    net_send(0,2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp   :derivimplicit
    i =  (0.001)*(g_r*(v-E_r) + g_s*(v-E_s))
}


DERIVATIVE states {		: solve eq for adaptation, relref variable
    g_s' = -g_s/tau_s
    g_r' = -g_r/tau_r
}

NET_RECEIVE (weight) {
    if (flag == 1) {        : beginning of spike
        state_discontinuity(g_s, g_s + q_s)
	state_discontinuity(g_r, g_r + q_r)
    } else if (flag == 2) { : watch membrane potential
        WATCH (v > vthresh) 1
    }
}