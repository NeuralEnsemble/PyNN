: Current-based synapse with exponential decay, for IF_curr_exp / current-based
: point neurons. An incoming event steps the synaptic current by `weight` (nA),
: which then decays exponentially with time constant `tau`. A positive weight
: injects depolarising (inward) current, matching PyNN's isyn convention, so the
: contributed membrane current is i = -isyn.
NEURON {
    POINT_PROCESS expsyn_curr
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
}

INITIAL {
    isyn = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = -isyn
}

DERIVATIVE state {
    isyn' = -isyn/tau
}

NET_RECEIVE(weight) {
    isyn = isyn + weight
}
