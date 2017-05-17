COMMENT
Implementation of a simple stochastic synapse (constant release probability)
as a "weight adjuster" (i.e. it sets the weight of the synapse to zero if
transmission fails).

Andrew Davison, UNIC, CNRS, 2016
ENDCOMMENT

NEURON {
    POINT_PROCESS SimpleStochasticWA
    RANGE p
    POINTER rng, wsyn
}

PARAMETER {
    p = 0.5   : probability that transmission succeeds
}

VERBATIM
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);

ENDVERBATIM

ASSIGNED {
    wsyn
    rng
}

NET_RECEIVE(w) {
    if (urand() < p) {
        wsyn = w
    } else {
        wsyn = 0.0
    }
}

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
