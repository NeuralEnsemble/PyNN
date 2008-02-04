: NetStim with Variable Rate
: Version 2 - self-timing
: Andrew P. Davison, UNIC, CNRS, July 2004

NEURON	{ 
  ARTIFICIAL_CELL NetStimVR2
  RANGE y, interval, noise, theta
  RANGE transform, prmtr, alpha, fthetastim
  GLOBAL tchange, thetastim
  GLOBAL Rmax, sigma, Rmin
}

PARAMETER {
	noise		= 1 <0,1> : amount of randomness (0.0 - 1.0)
        theta           = 0 <0,1> : location of this cell
	tchange         = 20 (ms) : time of next firing rate change
	thetastim       = 0 <0,1> : stimulus location
	Rmax            = 60 (/s) : peak firing rate
	Rmin            = 0  (/s) : min firing rate
	sigma           = 0.4     : width of tuning curve
	transform       = 0       : determines which transformation of thetastim
				  : to use 0=none 1=multiply 2=sin 3=cos
	prmtr           = 0       : parameter for the transformtion.
	alpha           = 1       
}

ASSIGNED {
	interval (ms)
	y
	event (ms)
	rate (/s)
	fthetastim
}

CONSTANT {
	PI = 3.141592654
}

PROCEDURE seed(x) {
	set_seed(x)
}

INITIAL {
	y = 0
	if (noise < 0) {
		noise = 0
	}
	if (noise > 1) {
		noise = 1
	}
	net_send(1e-12,4) : to make sure that ControlNSVR has
			  : time to set thetastim and tchange
}	

FUNCTION invl(mean (ms)) (ms) {
	if (mean <= 0.) {
		mean = .01 (ms) : I would worry if it were 0.
	}
	if (noise == 0) {
		invl = mean
	} else {
		invl = (1. - noise)*mean + noise*mean*exprand(1)
	}
}

FUNCTION tuning_curve(theta,theta0) (ms){
	LOCAL d
	d = theta - theta0
	if (d <= -0.5) { d = d + 1 }
	if (d > 0.5) { d = d - 1 }
	rate = alpha * ( (Rmax-Rmin) * exp( (cos(2*PI*d)-1)/(sigma*sigma) ) + Rmin )
	if (rate > 1e-9) {
		tuning_curve = (1000)/rate
	} else {
		tuning_curve = 1e12
	}
}

NET_RECEIVE (w) {
	if (flag == 4) { : from INITIAL
		if (transform == 1) {
			fthetastim = prmtr*thetastim
		} else if (transform == 2) {
			fthetastim = 0.5*(sin(2*PI*thetastim + prmtr) + 1)
		} else if (transform == 3) {
			fthetastim = thetastim*thetastim
		} else if (transform == 4) {
			fthetastim = asin(2*thetastim-1)/PI
		} else if (transform == 5) {
			fthetastim = 0.5*(sin(prmtr*2*PI*thetastim) + 1)
		} else {
			fthetastim = thetastim
		}
		interval = tuning_curve(theta,fthetastim)
		event = invl(interval) - interval*(1. - noise)
		if (event < 1e-12) {
			event = 1e-12
		}
		if (event < tchange) {
			net_send(event,1)
		}
		net_send(tchange,3)
	} else {
	if (flag == 3) { : change in mean rate
		if (transform == 1) {
			fthetastim = prmtr*thetastim
		} else if (transform == 2) {
			fthetastim = 0.5*(sin(2*PI*thetastim + prmtr) + 1)
		} else if (transform == 3) {
			fthetastim = thetastim*thetastim
		} else if (transform == 4) {
			fthetastim = asin(2*thetastim-1)/(2*PI)
		} else if (transform == 5) {
			fthetastim = 0.5*(sin(prmtr*2*PI*thetastim) + 1)
		} else {
			fthetastim = thetastim
		}	
		interval = tuning_curve(theta,fthetastim)
		event = t + invl(interval) - interval*(1. - noise)
		if (event < t) {
			event = t
		}
		if (event < tchange) {
			net_send(event-t,1)
		}
		net_send(tchange-t,3)
	} else {
	if (flag == 1) { : send the next spike
		y = 2
		net_event(t)
		event = event + invl(interval)
		if (event < tchange) {
			net_send(event - t, 1)
		}
		net_send(0.1, 2)
	} else {
	if (flag == 2) {
		y = 0
	}}}}
}

