: Based on netstim
: but with fixed duration rather than fixed number of spikes, and an interval
: that can safely be varied during the simulation
: Modified by Andrew Davison, UNIC, CNRS

NEURON	{ 
  ARTIFICIAL_CELL NetStimFD
  RANGE interval, start, duration
  RANGE noise
  THREADSAFE : only true if every instance has its own distinct Random
  POINTER donotuse
}

PARAMETER {
	interval	= 10 (ms) <1e-9,1e9> : time between spikes (msec)
	duration	= 100 (ms) <0,1e9>   : duration of firing (msec)
	start		= 50 (ms)	     : start of first spike
	noise		= 0 <0,1>	     : amount of randomness (0.0 - 1.0)
}

ASSIGNED {
	event (ms)
	on
	donotuse
	valid
}

PROCEDURE seed(x) {
	set_seed(x)
}

INITIAL {
        valid = 4
	on = 0 : off
	if (noise < 0) {
		noise = 0
	}
	if (noise > 1) {
		noise = 1
	}
	if (start >= 0 && duration > 0) {
		: randomize the first spike so on average it occurs at
		: start + noise*interval
                invl(interval) : for some reason, the first invl() call seems to give implausibly large values, so we discard it
		event = start + invl(interval) - interval*(1. - noise)
		: but not earlier than 0
		if (event < 0) {
			event = 0
		}
                if (event < start+duration) {
                        on = 1
                        net_send(event, 3)
                }
	}
}	

PROCEDURE init_sequence(t(ms)) {
	if (duration > 0) {
		on = 1
		event = 0
	}
}

FUNCTION invl(mean (ms)) (ms) {
	if (mean <= 0.0) {
		mean = 0.01 (ms) : I would worry if it were 0.0
	}
	if (noise == 0) {
		invl = mean
	}else{
		invl = (1.0 - noise)*mean + noise*mean*erand()
	}
}
VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

FUNCTION erand() {
VERBATIM
	if (_p_donotuse) {
		/*
		:Supports separate independent but reproducible streams for
		: each instance. However, the corresponding hoc Random
		: distribution MUST be set to Random.negexp(1)
		*/
		_lerand = nrn_random_pick(_p_donotuse);
	}else{
		/* only can be used in main thread */
		if (_nt != nrn_threads) {
hoc_execerror("multithread random in NetStim"," only via hoc Random");
		}
ENDVERBATIM
		: the old standby. Cannot use if reproducible parallel sim
		: independent of nhost or which host this instance is on
		: is desired, since each instance on this cpu draws from
		: the same stream
		erand = exprand(1)
VERBATIM
	}
ENDVERBATIM
}

PROCEDURE noiseFromRandom() {
VERBATIM
 {
	void** pv = (void**)(&_p_donotuse);
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (void*)0;
	}
 }
ENDVERBATIM
}

PROCEDURE next_invl() {
	if (duration > 0) {
		event = invl(interval)
	}
	if (t+event >= start+duration) {
		on = 0
	}
        :printf("t=%g, event=%g, t+event=%g, on=%g\n", t, event, t+event, on) 
}

NET_RECEIVE (w) {
	if (flag == 0) { : external event
                :printf("external event. w = %g\n", w)
		if (w > 0) { : turn on spike sequence
			: but not if a netsend is on the queue
			init_sequence(t)
			: randomize the first spike so on average it occurs at
			: noise*interval (most likely interval is always 0)
			next_invl()
			event = event - interval*(1.0 - noise)
			valid = valid + 1 : events with previous values of valid will be ignored.
			net_send(event, valid)
		}else if (w < 0) { : turn off spiking definitively
			on = 0
		}
	}
	if (flag == 3) { : from INITIAL
		if (on == 1) { : but ignore if turned off by external event
			init_sequence(t)
			net_send(0, valid)
                        :printf("init_sequence(%g)\n", t)
		}
	}
	if (flag == valid && on == 1) {
		net_event(t)
		next_invl()
		:printf("%g %g %g flag=%g valid=%g\n", t, interval, event, flag, valid)
		if (on == 1) {
			net_send(event, valid)
		}
	}
}

COMMENT
Presynaptic spike generator
---------------------------

This mechanism has been written to be able to use synapses in a single
neuron receiving various types of presynaptic trains.  This is a "fake"
presynaptic compartment containing a spike generator.  The trains
of spikes can be either periodic or noisy (Poisson-distributed)

Parameters;
   noise: 	between 0 (no noise-periodic) and 1 (fully noisy)
   interval: 	mean time between spikes (ms)
   [number: 	number of spikes (independent of noise)] - deleted
   duration:    duration of spiking (ms) - added

Written by Z. Mainen, modified by A. Destexhe, The Salk Institute

Modified by Michael Hines for use with CVode
The intrinsic bursting parameters have been removed since
generators can stimulate other generators to create complicated bursting
patterns with independent statistics (see below)

Modified by Michael Hines to use logical event style with NET_RECEIVE
This stimulator can also be triggered by an input event.
If the stimulator is in the on==0 state (no net_send events on queue)
 and receives a positive weight
event, then the stimulator changes to the on=1 state and goes through
its entire spike sequence before changing to the on=0 state. During
that time it ignores any positive weight events. If, in an on!=0 state,
the stimulator receives a negative weight event, the stimulator will
change to the on==0 state. In the on==0 state, it will ignore any ariving
net_send events. A change to the on==1 state immediately fires the first spike of
its sequence.

ENDCOMMENT

