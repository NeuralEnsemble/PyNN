: Generalized Integrate and Fire model defined in Pozzorini et al. PLOS Comp. Biol. 2015
: Filter for eta and gamma defined as linear combination of three exponential functions each.
:
: Implemented by Christian Roessert, EPFL/Blue Brain Project, 2016


NEURON {
    POINT_PROCESS GifCurrent
    RANGE Vr, Tref, Vt_star
    RANGE DV, lambda0
    RANGE tau_eta1, tau_eta2, tau_eta3, a_eta1, a_eta2, a_eta3
    RANGE tau_gamma1, tau_gamma2, tau_gamma3, a_gamma1, a_gamma2, a_gamma3
    RANGE i_eta, gamma_sum, v_t, verboseLevel, p_dontspike, rand
    RANGE e_spike, isrefrac
    POINTER rng
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (uS) = (microsiemens)
}

PARAMETER {
    Vr = -50 (mV)
    Tref = 4.0 (ms)
    Vt_star = -48 (mV)
    DV = 0.5 (mV)
    lambda0 = 1.0 (Hz)

    tau_eta1 = 1 (ms)
    tau_eta2 = 10. (ms)
    tau_eta3 = 100. (ms)
    a_eta1 = 1. (nA)
    a_eta2 = 1. (nA)
    a_eta3 = 1. (nA)

    tau_gamma1 = 1 (ms)
    tau_gamma2 = 10. (ms)
    tau_gamma3 = 100. (ms)
    a_gamma1 = 1. (mV)
    a_gamma2 = 1. (mV)
    a_gamma3 = 1. (mV)

    gon = 1e6 (uS)    : refractory clamp conductance
    e_spike = 0 (mV)  : spike height
}

COMMENT
The Verbatim block is needed to allow RNG.
ENDCOMMENT
VERBATIM
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

ASSIGNED {
    v (mV)
    i (nA)
    i_eta (nA)
    p_dontspike (1)
    lambda (Hz)
    irefrac (nA)
    rand (1)
    grefrac (uS)
    gamma_sum (mV)
    v_t (mV)
    verboseLevel (1)
    dt (ms)
    rng
    isrefrac (1) : is in refractory period
}

STATE {
    eta1  (nA)
    eta2  (nA)
    eta3  (nA)
    gamma1  (mV)
    gamma2  (mV)
    gamma3  (mV)
}

INITIAL {
    grefrac = 0
    eta1 = 0
    eta2 = 0
    eta3 = 0
    gamma1 = 0
    gamma2 = 0
    gamma3 = 0
    rand = urand()
    p_dontspike = 2
    isrefrac = 0
    net_send(0,4)
}

BREAKPOINT {
    SOLVE states METHOD cnexp :derivimplicit :euler

    i_eta = eta1 + eta2 + eta3

    gamma_sum = gamma1 + gamma2 + gamma3
    v_t = Vt_star + gamma_sum
    lambda = lambda0*exp( (v-v_t)/DV )
    if (isrefrac > 0) {
        p_dontspike = 2   : is in refractory period, make it impossible to trigger a spike
    } else {
        p_dontspike = exp(-lambda*(dt * (1e-3)))
    }

    irefrac = grefrac*(v-Vr)
    i = irefrac + i_eta

}

AFTER SOLVE {
    rand = urand()
}

DERIVATIVE states {		: solve spike frequency adaptation and spike triggered current kernels
    eta1' = -eta1/tau_eta1
    eta2' = -eta2/tau_eta2
    eta3' = -eta3/tau_eta3
    gamma1' = -gamma1/tau_gamma1
    gamma2' = -gamma2/tau_gamma2
    gamma3' = -gamma3/tau_gamma3
}

NET_RECEIVE (weight) {
    if (flag == 1) { : start spike next dt
        isrefrac = 1
        net_send(dt, 2)

        if( verboseLevel > 0 ) {
            printf("Next dt: spike, at time %g: rand=%g, p_dontspike=%g\n", t, rand, p_dontspike)
        }

    } else if (flag == 2) { : beginning of spike
        v = Vr
        grefrac = gon
        net_send(Tref-dt, 3)
        :net_event(t)

        : increase filters after spike
        eta1 = eta1 + a_eta1
        eta2 = eta2 + a_eta2
        eta3 = eta3 + a_eta3
        gamma1 = gamma1 + a_gamma1
        gamma2 = gamma2 + a_gamma2
        gamma3 = gamma3 + a_gamma3

        if( verboseLevel > 0 ) {
            printf("Start spike, at time %g: rand=%g, p_dontspike=%g\n", t, rand, p_dontspike)
        }

    } else if (flag == 3) { : end of refractory period
        v = Vr
        isrefrac = 0
        grefrac = 0

        if( verboseLevel > 0 ) {
            printf("End refrac, at time %g: rand=%g, p_dontspike=%g\n", t, rand, p_dontspike)
        }

    } else if (flag == 4) { : watch for spikes
        WATCH (rand>p_dontspike) 1
    }
}

PROCEDURE setRNG() {
VERBATIM
    {
        /**
         * This function takes a NEURON Random object declared in hoc and makes it usable by this mod file.
         * Note that this method is taken from Brett paper as used by netstim.hoc and netstim.mod
         * which points out that the Random must be in uniform(1) mode
         */
        void** pv = (void**)(&_p_rng);
        if( ifarg(1)) {
            *pv = nrn_random_arg(1);
        } else {
            *pv = (void*)0;
        }
    }
ENDVERBATIM
}

FUNCTION urand() {
VERBATIM
        double value;
        if (_p_rng) {
            /*
            :Supports separate independent but reproducible streams for
            : each instance. However, the corresponding hoc Random
            : distribution MUST be set to Random.negexp(1)
            */
            value = nrn_random_pick(_p_rng);
            //printf("random stream for this simulation = %lf\n",value);
            return value;
        }else{
ENDVERBATIM
            : the old standby. Cannot use if reproducible parallel sim
            : independent of nhost or which host this instance is on
            : is desired, since each instance on this cpu draws from
            : the same stream
            value = scop_random(1)
VERBATIM
        }
ENDVERBATIM
        urand = value
}

FUNCTION toggleVerbose() {
    verboseLevel = 1-verboseLevel
}
