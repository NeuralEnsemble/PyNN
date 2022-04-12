COMMENT
Implementation of the NEST quantal_stp_connection model for NEURON

Original NEST version by Marc-Oliver Gewaltig
Adapted to NMODL by Andrew Davison, UNIC, CNRS, 2016.
ENDCOMMENT

NEURON {
    POINT_PROCESS QuantalSTPWA
    RANGE tau_rec, tau_fac, U, u0, n
    POINTER wsyn, rng
}

PARAMETER {
    tau_rec = 800  (ms)         : time constant for depression
    tau_fac = 0    (ms)         : time constant for facilitation
    U       = 0.5  (1) <0, 1>   : maximal fraction of available resource
    u0      = 0.5  (1) <0, 1>   : initial available fraction of resources
	n       = 1                 : total number of release sites
}

ASSIGNED {
    u (1)        : available fraction of resources
    wsyn         : transmitted synaptic weight
    rng
}

INITIAL {
    u = u0
}

NET_RECEIVE(w, available, t_last (ms)) {
    : available - number of available release sites
    : t_last - time of the last spike

    LOCAL depleted, rv, p_decay, u_decay, n_release, i

    INITIAL{
        available = n
		t_last = -1e99
    }

    : Compute the decay factors, based on the time since the last spike.
    p_decay = exp(-(t - t_last)/tau_rec)
    if (tau_fac < 1e-10) {
        u_decay = 0.0
	} else {
		u_decay = exp( -(t - t_last)/tau_fac)
    }

    : Compute release probability
    u = U + u*(1 - U)*u_decay

    : Compute number of sites that recovered during the interval.
    depleted = n - available
    while (depleted > 0) {
        rv = urand()
        if (rv < (1 - p_decay)) {
            available = available + 1
		}
        depleted = depleted - 1
    }

    : Compute number of released sites
    n_release = 0
    i = available
    while (i > 0) {
        rv = urand()
		if (rv < u) {
            n_release = n_release + 1
        }
        i = i - 1
    }

    if (n_release > 0) {
		wsyn = n_release/n * w
        available = available - n_release
    } else {
		wsyn = 0
    }
	t_last = t
}

VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

PROCEDURE setRNG() {
    : This function takes a NEURON Random object declared in hoc and makes it usable by this mod file
    : The Random must be in uniform(1) mode
VERBATIM
    {
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
        } else {
ENDVERBATIM
            value = scop_random(1)
VERBATIM
        }
ENDVERBATIM
        urand = value
}
