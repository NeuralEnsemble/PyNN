NEURON {
    SUFFIX kdr
    USEION k  READ ek  WRITE ik
    RANGE gkbar, q10
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gkbar  =   0.036  (S/cm2)
    celsius           (degC)
}

STATE { n }

ASSIGNED { q10 }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gk, n2

    n2 = n*n
    gk = gkbar*n2*n2
    ik  = gk*(v - ek)
}

INITIAL {
    LOCAL alpha, beta

    q10 = 3^(0.1*celsius - 0.63)

    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)
    n     = alpha/(alpha + beta)
}

DERIVATIVE states {
    LOCAL alpha, beta

    : potassium activation system
    alpha = n_alpha(v)
    beta  = n_beta(v)
    n'    = (alpha - n*(alpha + beta))*q10
}

FUNCTION n_alpha(v) { n_alpha = 0.1*exprelr(-0.1*v - 5.5) }
FUNCTION n_beta(v)  { n_beta  = 0.125*exp(-0.0125*v - 0.8125) }