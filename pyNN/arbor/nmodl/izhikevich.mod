: Izhikevich (2003) quadratic integrate-and-fire neuron, for building the PyNN
: Izhikevich cell type as an Arbor cable cell:
:     dv/dt = 0.04 v^2 + 5 v + 140 - u + I
:     du/dt = a (b v - u)
: with reset, when v reaches vthresh: v -> c, u -> u + d.
:
: v is integrated by Arbor's cable solver: with the cell's absolute capacitance set
: equal to Cm, this POINT_PROCESS supplies i = -Cm(0.04 v^2 + 5 v + 140 - u) so that
: Cm dv/dt = -i reproduces the Izhikevich dv/dt (the injected current I enters via a
: separate iclamp for i_offset, contributing I/Cm as in the NEURON backend). There
: is no leak mechanism. u is a STATE solved by cnexp.
:
: Arbor mechanisms cannot write v, so the reset (v -> c) is done as in lif.mod: the
: threshold_detector (set to vthresh) delivers a POST_EVENT that starts a very short
: countdown during which a strong clamp holds v at c; the countdown t_reset is tiny
: (<= one timestep) so there is no artificial refractory period, only a one-step
: reset. u += d is applied in the POST_EVENT. The clamp is gated by the
: voltage-independent 0/1 multiplier `clamp` so modcc derives the conductance
: correctly (see lif.mod).
NEURON {
    POINT_PROCESS izhikevich
    RANGE a, b, c, d, vthresh, Cm, uinit, t_reset, g_reset
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (nF) = (nanofarad)
    (uS) = (microsiemens)
    (ms) = (millisecond)
}

PARAMETER {
    a       = 0.02   (/ms)
    b       = 0.2    (/ms)
    c       = -65    (mV)      : reset potential
    d       = 2      (mV/ms)   : reset increment of u
    vthresh = 30     (mV)      : spike / reset threshold
    Cm      = 0.001  (nF)      : capacitance (matches the cell's absolute cm)
    uinit   = -14    (mV/ms)
    t_reset = 0.001  (ms)      : clamp duration (<= dt: a one-step reset)
    g_reset = 1000   (uS)
}

STATE {
    u      (mV/ms)
    refrac (ms)
}

ASSIGNED { clamp }

INITIAL {
    u = uinit
    refrac = 0
    clamp = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    clamp = 0
    if (refrac > 0) { clamp = 1 }
    i = clamp * g_reset * (v - c)
        + (1 - clamp) * (-Cm * (0.04 * v * v + 5 * v + 140 - u))
}

DERIVATIVE states {
    u' = a * (b * v - u)
    refrac' = -1
}

POST_EVENT(time) {
    refrac = t_reset
    u = u + d
}
