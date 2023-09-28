NEURON {
    SUFFIX pas
    NONSPECIFIC_CURRENT i
    RANGE g
    GLOBAL e
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

INITIAL {}

PARAMETER {
    g = .001 (S/cm2)
    e = -70  (mV) : Taken from nrn
}

BREAKPOINT {
    i = g*(v - e)
}
