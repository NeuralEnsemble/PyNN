COMMENT

Spike generator following a Poisson process with a refractory period.

Parameters:
    rate:        Mean spike frequency (Hz)
    tau_refrac:  Minimum time between spikes (ms)
    start:       Start time (ms)
    duration:    Duration of spike sequence (ms)

Author: Andrew P. Davison, UNIC, CNRS

ENDCOMMENT

NEURON  {
    ARTIFICIAL_CELL PoissonStimRefractory
    RANGE rate, tau_refrac, start, duration
}

PARAMETER {
    rate = 1.0 (Hz)
    tau_refrac = 0.0 (ms)
    start = 1 (ms)
    duration = 1000 (ms)
}

ASSIGNED {
    event (ms)
    on
    end (ms)
}

PROCEDURE seed(x) {
    set_seed(x)
}

INITIAL {
    on = 0
    if (start >= 0) {
        net_send(event, 2)
    }
}

NET_RECEIVE (w) {
    LOCAL mean_poisson_interval
    if (flag == 2) { : from INITIAL
        if (on == 0) {
            on = 1
            event = t
            end = t + 1e-6 + duration
            net_send(0, 1)
        }
    }
    if (flag == 1 && on == 1) {
        net_event(t)
        mean_poisson_interval = 1000.0/rate - tau_refrac
        event = event + tau_refrac + mean_poisson_interval * exprand(1)
        if (event > end) {
            on = 0
        }
        if (on == 1) {
            net_send(event - t, 1)
        }
    }
}
