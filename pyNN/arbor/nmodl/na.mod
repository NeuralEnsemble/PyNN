NEURON {
    SUFFIX na
    USEION na READ ena WRITE ina
    RANGE gnabar, q10
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar =   0.12   (S/cm2)
    celsius           (degC)
}

STATE { m h }

ASSIGNED { q10 }

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL gna

    gna = gnabar*m*m*m*h
    ina = gna*(v - ena)
}

INITIAL {
    LOCAL alpha, beta

    q10 = 3^(0.1*celsius - 0.63)

    : sodium activation system
    alpha = m_alpha(v)
    beta  = m_beta(v)
    m     = alpha/(alpha + beta)

    : sodium inactivation system
    alpha = h_alpha(v)
    beta  = h_beta(v)
    h     = alpha/(alpha + beta)

}

DERIVATIVE states {
    LOCAL alpha, beta

    : sodium activation system
    alpha = m_alpha(v)
    beta  = m_beta(v)
    m'    = (alpha - m*(alpha + beta))*q10

    : sodium inactivation system
    alpha = h_alpha(v)
    beta  = h_beta(v)
    h'    = (alpha - h*(alpha + beta))*q10

}

FUNCTION m_alpha(v) { m_alpha = exprelr(-0.1*v - 4.0) }
FUNCTION h_alpha(v) { h_alpha = 0.07*exp(-0.05*v - 3.25) }

FUNCTION m_beta(v)  { m_beta  = 4.0*exp(-(v + 65.0)/18.0) }
FUNCTION h_beta(v)  { h_beta  = 1.0/(exp(-0.1*v - 3.5) + 1.0) }
