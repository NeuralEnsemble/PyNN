: Integrate-and-fire reset mechanism, for building point neurons as Arbor cable
: cells (a single-compartment cell whose sub-threshold leak is a separate `pas`
: mechanism and whose network spike is emitted by the cell's threshold_detector).
:
: This mechanism performs only the post-spike reset and refractory clamp: when the
: cell spikes, Arbor delivers a POST_EVENT here, which reloads a refractory
: countdown `refrac` (in ms). While the countdown is positive the mechanism injects
: a strong current g_reset*(v - v_reset) that holds the membrane at v_reset; when it
: is not, it contributes nothing. Arbor NMODL exposes neither absolute time nor
: WATCH (NEURON's adexp.mod approach), hence the countdown-plus-POST_EVENT design.
:
: The clamp is gated by a voltage-independent multiplier `clamp` (0/1) rather than
: writing the piecewise current directly, so that the automatically-derived
: conductance di/dv is exactly clamp*g_reset (0 when not refractory); guarding the
: current expression with a plain `if (refrac > 0)` instead makes modcc leak a
: spurious conductance even when the current is zero.
NEURON {
    POINT_PROCESS lif
    RANGE v_reset, t_ref, g_reset
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
    (ms) = (millisecond)
}

PARAMETER {
    v_reset = -65  (mV)
    t_ref   = 0.1  (ms)
    g_reset = 1000 (uS)   : large clamp conductance (tau_clamp = C_m/g_reset << dt)
}

STATE { refrac (ms) }

ASSIGNED { clamp }

INITIAL {
    refrac = 0
    clamp = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    clamp = 0
    if (refrac > 0) { clamp = 1 }
    i = clamp * g_reset * (v - v_reset)
}

DERIVATIVE states {
    refrac' = -1
}

POST_EVENT(time) {
    refrac = t_ref
}
