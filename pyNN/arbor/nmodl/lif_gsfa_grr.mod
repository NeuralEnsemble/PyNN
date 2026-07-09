: Integrate-and-fire reset with conductance-based spike-frequency adaptation (g_s)
: and a conductance-based relative-refractory mechanism (g_r), for building
: IF_cond_exp_gsfa_grr as an Arbor cable cell (Muller et al. 2007). Combines the
: lif.mod reset (refractory countdown + clamp, driven by the threshold_detector via
: POST_EVENT) with two spike-triggered conductances that are incremented by q_s/q_r
: on each spike and decay with time constants tau_s/tau_r, pulling the membrane
: towards E_s/E_r. The sub-threshold leak is a separate `pas` mechanism, as for lif.
:
: The adaptation current is 0.001 * (g_s*(v-E_s) + g_r*(v-E_r)): with g in nS and v
: in mV, g*(v-E) is in pA, and the 0.001 factor converts it to nA (matching the
: NEURON backend's gsfa_grr.mod). The refractory clamp is gated by the
: voltage-independent 0/1 multiplier `clamp` so modcc derives the conductance
: correctly (see lif.mod).
NEURON {
    POINT_PROCESS lif_gsfa_grr
    RANGE v_reset, t_ref, g_reset, E_s, E_r, tau_s, tau_r, q_s, q_r
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
    (nS) = (nanosiemens)
    (ms) = (millisecond)
}

PARAMETER {
    v_reset = -65   (mV)
    t_ref   = 0.1   (ms)
    g_reset = 1000  (uS)
    E_s     = -75   (mV)
    E_r     = -75   (mV)
    tau_s   = 100   (ms)
    tau_r   = 2     (ms)
    q_s     = 15    (nS)
    q_r     = 3000  (nS)
}

STATE {
    refrac (ms)
    g_s    (nS)
    g_r    (nS)
}

ASSIGNED { clamp }

INITIAL {
    refrac = 0
    g_s = 0
    g_r = 0
    clamp = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    clamp = 0
    if (refrac > 0) { clamp = 1 }
    i = clamp * g_reset * (v - v_reset)
        + (0.001) * (g_s * (v - E_s) + g_r * (v - E_r))
}

DERIVATIVE states {
    refrac' = -1
    g_s' = -g_s/tau_s
    g_r' = -g_r/tau_r
}

POST_EVENT(time) {
    refrac = t_ref
    g_s = g_s + q_s
    g_r = g_r + q_r
}
