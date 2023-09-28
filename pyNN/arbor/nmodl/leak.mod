NEURON {
    SUFFIX leak
    NONSPECIFIC_CURRENT il
    RANGE gl
    GLOBAL el
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gl     =   0.0003 (S/cm2)
    el     = -54.3    (mV)
}

BREAKPOINT {
    il  = gl*(v - el)
}
