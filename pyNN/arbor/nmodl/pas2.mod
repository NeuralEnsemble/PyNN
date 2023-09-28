NEURON {
    SUFFIX pas2
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

INITIAL {}

PARAMETER {
    g = .001 (S/cm2)
    e = -70  (mV)
}

BREAKPOINT {
    i = g*(v - e)
}
