COMMENT
Implementation of the Tsodyks-Markram mechanism for synaptic depression and
facilitation as a "weight adjuster"
Andrew Davison, UNIC, CNRS, 2013
ENDCOMMENT

NEURON {
    POINT_PROCESS TsodyksMarkramWA
    RANGE tau_rec, tau_facil, U, u0, tau_syn
    POINTER wsyn
}

PARAMETER {
    tau_rec   = 100  (ms) <1e-9, 1e9>
    tau_facil = 1000 (ms) <0, 1e9>
    U         = 0.04 (1)  <0, 1>
    u0        = 0    (1)  <0, 1>
    tau_syn   = 2    (ms) <1e-9, 1e9>  : should be set to be the same as the receiving synapse
}

ASSIGNED {
    x
    y
    z
    u
    t_last (ms)
    wsyn
}

INITIAL {
    y = 0
    z = 0
    u = u0
    t_last = -1e99
}

NET_RECEIVE(w) {
    INITIAL {
        t_last = t
    }
    z = z*exp(-(t - t_last)/tau_rec)
    z = z + y*(exp(-(t - t_last)/tau_syn) - exp(-(t - t_last)/tau_rec)) / ((tau_syn/tau_rec) - 1)
    y = y*exp(-(t - t_last)/tau_syn)
    x = 1 - y - z
    if (tau_facil > 0) {
        u = u*exp(-(t - t_last)/tau_facil)
        u = u + U*(1-u)
    } else {
        u = U
    }
    wsyn = w*x*u
    y = y + x*u
    :printf("U=%g  tau_rec=%g  tau_facil=%g  tau_syn=%g  w=%g\n", U, tau_rec, tau_facil, tau_syn, w)
    :printf("%g\t%g\t%g\t%g\t%g\t%g\t%g\n", t, t - t_last, y, z, u, x, wsyn)
    t_last = t
}
