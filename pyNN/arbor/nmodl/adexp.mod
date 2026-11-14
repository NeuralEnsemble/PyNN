: Adaptive-exponential (Brette-Gerstner) dynamics for building AdExp / EIF point
: neurons as Arbor cable cells. Like lif.mod this is a POINT_PROCESS placed at the
: soma whose reset is driven by the cell's threshold_detector via POST_EVENT; it
: additionally contributes the exponential spike-generating current and the
: adaptation current w. The sub-threshold leak -GL*(v - EL) is a separate `pas`
: density mechanism (with the same GL), exactly as for lif.mod.
:
: Membrane dynamics realised (with C dv/dt = -sum of mechanism currents):
:     C dv/dt = -GL(v-EL) + GL*delta*exp((v-vthresh)/delta) - w + I
:     tau_w dw/dt = a*(v - EL) - w
: so this mechanism contributes i = GL*delta term (as iexp, negative/inward) + w.
: On a spike (detector crosses v_spike) POST_EVENT starts a refractory countdown and
: increments w by b; while refractory a strong clamp holds v at v_reset (and the
: exponential/adaptation currents are gated off), matching lif.mod. The clamp is
: gated by a voltage-independent 0/1 multiplier so modcc derives the conductance
: correctly (see lif.mod).
NEURON {
    POINT_PROCESS adexp
    RANGE v_reset, t_ref, g_reset, GL, delta, vthresh, a, b, tau_w, EL
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
    (ms) = (millisecond)
}

PARAMETER {
    v_reset = -70.6  (mV)
    t_ref   = 0.1    (ms)
    g_reset = 1000   (uS)
    GL      = 0.03   (uS)   : leak conductance (= C_m/tau_m; matches the pas leak)
    delta   = 2      (mV)   : steepness of the exponential
    vthresh = -50.4  (mV)   : exponential (soft) threshold V_T
    a       = 0.004  (uS)   : subthreshold adaptation conductance
    b       = 0.0805 (nA)   : spike-triggered adaptation increment
    tau_w   = 144    (ms)   : adaptation time constant
    EL      = -70.6  (mV)   : leak reversal (= v_rest)
}

STATE {
    refrac (ms)
    w      (nA)
}

ASSIGNED { clamp  iexp (nA) }

INITIAL {
    refrac = 0
    w = 0
    clamp = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    clamp = 0
    if (refrac > 0) { clamp = 1 }
    iexp = exp_current(v)
    i = clamp * g_reset * (v - v_reset) + (1 - clamp) * (iexp + w)
}

DERIVATIVE states {
    refrac' = -1
    w' = (a * (v - EL) - w) / tau_w
}

FUNCTION exp_current(v (mV)) (nA) {
    LOCAL arg
    arg = (v - vthresh) / delta
    if (arg > 50) { arg = 50 }   : guard against overflow before the reset clamps v
    exp_current = -GL * delta * exp(arg)
}

POST_EVENT(time) {
    refrac = t_ref
    w = w + b
}
