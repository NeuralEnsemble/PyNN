: Implements Song-Abbott training sequence
: controls NetStimVRs
: Andrew P. Davison, UNIC, CNRS, July 2004

NEURON {
  ARTIFICIAL_CELL ControlNSVR2
  RANGE tau_corr
  POINTER thetastim
  POINTER tchange
}

PROCEDURE seed(x) {
	set_seed(x)
}

ASSIGNED {
	tchange (ms)
	thetastim
}
	
PARAMETER {
	tau_corr = 20 (ms)
}

INITIAL {
	tchange = tau_corr*exprand(1)
	net_send(tchange-1e-12,1)
}

NET_RECEIVE (w) {
	if (flag == 1) {
		tchange = t+1e-12 + tau_corr*exprand(1)
		thetastim = scop_random()
		net_send(tchange-t-1e-12,1)
	} 
}	