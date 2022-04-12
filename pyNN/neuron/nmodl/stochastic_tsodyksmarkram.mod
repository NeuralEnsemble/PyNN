COMMENT
Implementation of the stochastic Tsodyks-Markram mechanism for synaptic depression and
facilitation as a "weight adjuster"

cf Fuhrmann et al. 2002
The algorithm is as in ProbGABAAB_EMS.mod from the Blue Brain Project.

Andrew Davison, UNIC, CNRS, 2016.
ENDCOMMENT

NEURON {
    POINT_PROCESS StochasticTsodyksMarkramWA
    RANGE tau_rec, tau_facil, U, u0
    POINTER wsyn, rng
}

PARAMETER {
    tau_rec   = 100  (ms) <1e-9, 1e9>
    tau_facil = 1000 (ms) <0, 1e9>
    U         = 0.04 (1)  <0, 1>
    u0        = 0    (1)  <0, 1>
}

ASSIGNED {
    u (1)        : release probability
    t_last (ms)  : time of the last spike
    wsyn         : transmitted synaptic weight
    R (1)        : recovered state {0=unrecovered, 1=recovered}
    rng
}

INITIAL {
    u = u0
    t_last = -1e99
	R = 1
}

NET_RECEIVE(w, p_surv, t_surv) {
    : p_surv - survival probability of unrecovered state
    : t_surv - time since last evaluation of survival
    LOCAL result
    INITIAL{
		t_last = t
    }

    if (w > 0) {
        :printf("START tau_facil=%-4g  tau_rec=%-4g  U=%-4.2g  time=%g  p_surv=%-5.3g  t_surv=%4.1f  t_last=%4.1f  u=%-5.3g  R=%g  wsyn=%g\n", tau_facil, tau_rec, U, t, p_surv, t_surv, t_last, u, R, wsyn)
        : calculation of u
        if (tau_facil > 0) {
            u = u*exp(-(t - t_last)/tau_facil)
            u = u + U*(1-u)
        } else {
            u = U
        }
        t_last = t

        : check for recovery
        if (R == 0) {
			wsyn = 0
            : probability of survival of unrecovered state based on Poisson recovery with rate 1/tau_rec
            p_surv = exp(-(t - t_surv)/tau_rec)
            result = urand()
            if (result > p_surv) {
                R = 1        : recovered
				:printf("recovered\n")
            } else {
                t_surv = t   : failed to recover
            }
        }

        : check for release
        if (R == 1) {
            result = urand()
            if (result < u) {    : release
                wsyn = w
                R = 0
                t_surv = t
				:printf("release\n")
            } else {
                wsyn = 0
            }
        }
        :printf("END   tau_facil=%-4g  tau_rec=%-4g  U=%-4.2g  time=%g  p_surv=%-5.3g  t_surv=%4.1f  t_last=%4.1f  u=%-5.3g  R=%g  wsyn=%g\n\n", tau_facil, tau_rec, U, t, p_surv, t_surv, t_last, u, R, wsyn)
    }
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
