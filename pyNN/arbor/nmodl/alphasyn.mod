: Conductance-based synapse with an alpha-function time course, for IF_cond_alpha
: and conductance-based point neurons. An incoming event of weight `w` (uS) produces
: a conductance g(t) = w * (t/tau) * exp(1 - t/tau), peaking at `w` a time `tau`
: after the event; the membrane current is i = g*(v - e).
:
: The alpha shape is the impulse response of the coupled linear system
:     g' = z - g/tau
:     z' = -z/tau
: with each event adding w*exp(1)/tau to z (so that the peak of g equals w). Unlike
: the NEURON backend's alphasyn.mod this needs no spike queue. The system is a
: non-diagonalisable Jordan block, so `cnexp` cannot solve it; `sparse` (Arbor's
: implicit/backward-Euler solver) is used instead, which is first-order accurate --
: the alpha peak is slightly under-estimated at large dt (a few % of the synaptic
: current for dt ~ tau), but the downstream membrane response matches the exact
: alpha to <1% for dt <= 0.05 ms.
NEURON {
    POINT_PROCESS alphasyn
    RANGE tau, e
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (ms) = (millisecond)
}

PARAMETER {
    tau = 5.0 (ms)
    e   = 0   (mV)
}

STATE {
    g (uS)
    z (uS/ms)
}

INITIAL {
    g = 0
    z = 0
}

BREAKPOINT {
    SOLVE state METHOD sparse
    i = g * (v - e)
}

DERIVATIVE state {
    g' = z - g/tau
    z' = -z/tau
}

NET_RECEIVE(weight (uS)) {
    z = z + weight * exp(1) / tau
}
