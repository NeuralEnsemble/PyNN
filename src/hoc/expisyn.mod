NEURON {
	POINT_PROCESS ExpISyn
	RANGE tau, i, k
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	k = 1
}

ASSIGNED {
	v (mV)
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

NET_RECEIVE(weight (uS)) {
	state_discontinuity(i, i - k*weight)
}

