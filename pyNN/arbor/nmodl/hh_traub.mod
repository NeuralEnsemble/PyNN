: Traub-modified Hodgkin-Huxley channels, for building HH_cond_exp as an Arbor
: single-compartment cable cell. A near-verbatim port of the NEURON backend's
: hh_traub.mod: the TABLE statement is dropped (Arbor's modcc has no TABLE; the
: rates are computed exactly each step) and the rate "trap" is written with Arbor's
: exprelr (vtrap(x,y) = y*exprelr(x/y) = x/(exp(x/y)-1), singularity-safe). The
: NEST-compatible initial state m = h = n = 0 is kept. Reversal potentials for Na/K
: come from the ions (set per cell); the leak is a NONSPECIFIC_CURRENT.
NEURON {
    SUFFIX hh_traub
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar, gkbar, gl, el, vT
}

UNITS {
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gnabar = 0.02    (S/cm2)
    gkbar  = 0.006   (S/cm2)
    gl     = 0.00001 (S/cm2)
    el     = -60.0   (mV)
    vT     = -63.0   (mV)
}

STATE { m h n }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m * m * m * h * (v - ena)
    ik  = gkbar * n * n * n * n * (v - ek)
    il  = gl * (v - el)
}

INITIAL {
    : for compatibility with NEST, start from m = h = n = 0
    m = 0
    h = 0
    n = 0
}

DERIVATIVE states {
    LOCAL u, alpha, beta
    u = v - vT
    : sodium activation
    alpha = 0.32 * vtrap(13 - u, 4)
    beta  = 0.28 * vtrap(u - 40, 5)
    m' = alpha - m * (alpha + beta)
    : sodium inactivation
    alpha = 0.128 * exp((17 - u) / 18)
    beta  = 4 / (exp((40 - u) / 5) + 1)
    h' = alpha - h * (alpha + beta)
    : potassium activation
    alpha = 0.032 * vtrap(15 - u, 5)
    beta  = 0.5 * exp((10 - u) / 40)
    n' = alpha - n * (alpha + beta)
}

FUNCTION vtrap(x, y) {
    vtrap = y * exprelr(x / y)
}
