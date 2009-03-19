TITLE Brette-Gerstner-Izhikevich IF model

COMMENT
-----------------------------------------------------------------------------

Integrate and fire model based on Brette-Gerstner-Izhikevich model. 
This model consists of the two-variable integrate-and-fire (IF) model
proposed by Izhikevich (2004):

  Izhikevich EM. Which model to use for cortical spiking neurons?
  IEEE Trans. Neural Networks. 15: 1063-1070, 2004.

This model was modified to include an exponential non-linearity
around spike threshold, based on the exponential IF model of 
Fourcaud-Trocme et al. (2003):

  Fourcaud-Trocme N, Hansel D, van Vreeswijk C and Brunel N.
  How spike generation mechanisms determine the neuronal response to
  fluctuating inputs.  J. Neurosci. 23: 11628-11640, 2003.  

These two models were combined by Brette and Gerstner (2005):

  Brette R and Gerstner W. Adaptive exponential integrate-and-fire
  model as an effective description of neuronal activity. 
  J. Neurophysiol. 94: 3637-3642, 2005.

(see this paper for details)

The present implementation implements reset and the absolute refractory period
using a strong hyperpolarizing (gkd) conductance which clamps the membrane at
reset value for some refractory time. The above-threshold trajectory of the
membrane potential during a spike is not modelled.

This spike mechanism is implemented as a regular NEURON membrane
mechanism, and thus can be used in any compartment.  For faster
versions of the IF model, see the IntFire mechanisms in NEURON.

Parameters of the Brette-Gerstner-Izhikevich model:

	a (uS)     : level of adaptation
	b (nA)     : increment of adaptation at each spike
	tau_w (ms) : time constant of adaptation
	EL (mV)    : leak reversal (must be equal to the e_pas)
	GL (S/cm2) : leak conductance (must be equal to g_pas)
	delta (mV) : steepness of exponential approach to threshold
	surf (um2) : cell area

Additional parameters for spike generation:

	Ref (ms)   : refractory period (Vm is clamped at reset value)
	Vtop (mV)  : peak value of the spike
        Vspike (mV) : spike-detection threshold
	Vbot (mV)  : reset value
	gkd (S/cm2): gKd for reset

model 3: simpler IF mechanism, no spike duration
model 4: store spike times in vector

Original author: Alain Destexhe

model 5: removed Na current, removed spike-time storage

Modified by: Andrew Davison

-----------------------------------------------------------------------------
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX IF_BG5
	NONSPECIFIC_CURRENT i
	RANGE w, w_init
	RANGE a, b, tau_w, EL, GL, delta, surf
	RANGE Vtr, Ref, Vbot, Vtop, Vspike
	RANGE reset, gkd
        RANGE debug
}

UNITS {
	(mA) 	= (milliamp)
	(nA) 	= (nanoamp)
	(mV) 	= (millivolt)
	(umho) 	= (micromho)
	(um) 	= (micrometers)
	(uS) 	= (microsiemens)
}

PARAMETER {
	dt              (ms)

: Brette-Gerstner-Izhikevich parameters : 

	a 	= 0.027	(uS)		: level of adaptation
	b	= 0.019 (nA)		: increment of adaptation
	tau_w	= 120 	(ms)		: time constant of adaptation
	EL	= -80 	(mV)		: leak reversal (must be equal to e_pas)
	GL	= 0.0001 (mho/cm2)	: leak conductance (must be equal to g_pas)
	delta	= 2 	(mV)		: steepness of exponential approach to threshold
	surf 	= 10000	(um2) 		: cell area

: spike generation parameters :
	Vtr 	= -50	(mV) 		: voltage threshold for spike
	Ref	= 2	(ms) 		: refractory period (Vm is clamped at reset value)
	Vtop	= 50	(mV)		: peak value of spike
        Vspike  = 0     (mV)            : spike detection threshold
	Vbot	= -85	(mV)		: reset value
	gkd	= 10000	(mho/cm2)	: very strong gKd for reset
        w_init  = 0     (nA)
        
        debug = 0
}


ASSIGNED {
	v		(mV)		: membrane potential
	i		(mA/cm2)	: membrane current

	reset	(ms)			: reset counter
}

STATE {
	w   		(nA)
}

INITIAL {
	reset = -1
	w = w_init
}

BREAKPOINT {
	SOLVE states METHOD derivimplicit
	i = - GL * delta * exp((v-Vtr)/delta) + (100) * w/surf
	fire()
}


DERIVATIVE states {		: solve eq for adaptation variable
	w'=(a*(v-EL)-w)/tau_w
}



PROCEDURE fire() {

	reset = reset - 0.5*dt : fire() is called twice per timestep

	if (reset>0) {   			: inside the reset ?
		i = gkd * (v-Vbot)		: hyp current
                if (debug > 0) { printf("reset %f\t%f\t%f\t%f\t%f\t%f\t%f\n",t,v,w,i,reset,(v-Vtr)/delta,exp((v-Vtr)/delta)) }
	} else if (v>Vspike) {		        : passing threshold ?
                i = gkd * (v-Vbot)		: hyp current
		w = w + b			: increment adaptation var
                reset = Ref			: initiate reset
                if (debug > 0) { printf("spike %f\t%f\t%f\t%f\t%f\t%f\t%f\n",t,v,w,i,reset,(v-Vtr)/delta,exp((v-Vtr)/delta)) }
  	}
}

