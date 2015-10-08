NEURON {
	POINT_PROCESS ExpISyn
	RANGE tau, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
}

STATE {
	i (nA)
}

INITIAL {
	i = 0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
}

DERIVATIVE state {
	i' = -i/tau
}

NET_RECEIVE(weight (nA)) {
        :printf("t = %f, weight = %f\n", t, weight)
	i = i - weight
}

