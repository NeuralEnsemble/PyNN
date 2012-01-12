TITLE Alpha-function synaptic current, with NET_RECEIVE

COMMENT
This model works with variable time-step methods (although it may not
be very accurate) but at the expense of having to maintain the queues
of spike times and weights.

Andrew P. Davison, UNIC, CNRS, May 2006
ENDCOMMENT

DEFINE MAX_SPIKES 1000
DEFINE CUTOFF 20

NEURON {
	POINT_PROCESS AlphaISyn
	RANGE tau, i, q
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	tau = 5 (ms) <1e-9,1e9>

}

ASSIGNED {
	i (nA)
	q
	quiet
	onset_times[MAX_SPIKES] (ms)
	weight_list[MAX_SPIKES] (nA)
}

INITIAL {
	i  = 0
	q  = 0 : queue index
	quiet = 0
}

BREAKPOINT {
	LOCAL k, expired_spikes, x
	i = 0
	expired_spikes = 0
	FROM k=0 TO q-1 {
		x = (t - onset_times[k])/tau
		if (x > CUTOFF) {
			expired_spikes = expired_spikes + 1
		} else {
			i = i - weight_list[k] * alpha(x)
		}
	}
	update_queue(expired_spikes)
}

FUNCTION update_queue(n) {
	LOCAL k
	:if (n > 0) { printf("Queue changed. t = %4.2f onset_times=[",t) }
	FROM k=0 TO q-n-1 {
		onset_times[k] = onset_times[k+n]
		weight_list[k] = weight_list[k+n]
		:if (n > 0) { printf("%4.2f ",onset_times[k]) }
	}
	:if (n > 0) { printf("]\n") }
	q = q-n
}

FUNCTION alpha(x) {
	if (x < 0) {
		alpha = 0
	} else {
		alpha = x * exp(1 - x)
	}
}

NET_RECEIVE(weight (nA)) {
	onset_times[q] = t
	weight_list[q] = weight
        :printf("t = %f, weight = %f\n", t, weight)
	if (q >= MAX_SPIKES-1) {
		if (!quiet) {
			printf("Error in AlphaSynI. Spike queue is full\n")
			quiet = 1
		}
	} else {
		q = q + 1
	}
}
