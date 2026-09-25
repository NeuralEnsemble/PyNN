: Current-based synapse with an alpha-function time course, for IF_curr_alpha and
: current-based point neurons. An incoming event of weight `w` (nA) produces a
: synaptic current isyn(t) = w * (t/tau) * exp(1 - t/tau), peaking at `w` a time
: `tau` after the event. A positive weight injects depolarising (inward) current,
: matching PyNN's convention, so the contributed membrane current is i = -isyn.
:
: Same coupled-ODE realisation of the alpha shape as alphasyn.mod (no spike queue);
: solved with `sparse` (implicit) because the Jordan-block system is not diagonal.
NEURON {
    POINT_PROCESS alphasyn_curr
    RANGE tau
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (ms) = (millisecond)
}

PARAMETER {
    tau = 5.0 (ms)
}

STATE {
    isyn (nA)
    z    (nA/ms)
}

INITIAL {
    isyn = 0
    z = 0
}

BREAKPOINT {
    SOLVE state METHOD sparse
    i = -isyn
}

DERIVATIVE state {
    isyn' = z - isyn/tau
    z' = -z/tau
}

NET_RECEIVE(weight (nA)) {
    z = z + weight * exp(1) / tau
}
