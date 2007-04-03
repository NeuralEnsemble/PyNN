COMMENT

Assuming we have a Vector of spike times, how can we use that as a
stimulus to a cell? In analogy to the Vector.record/Vector.play methods,
one might expect that NetCon.play(Vector) would do the job. 
Unfortunately, at the time of version 5.1, perusing the list of NetCon
methods in the help system doesn't show such a method and the closest
candidate, NetCon.event, cannot take a Vector as an argument. 

Until NEURON has this functionality built-in, the best
way to send a Vector stream of events through a NetCon object is to
pass a Vector to a PointProcess model description
similar to NetStim.mod and use
that as a source for NetCons.

If the model is called VecStim, then we'd like its usage to be
as similar to NetStim as possible, ie.
	objref vs
	vs = new VecStim(.5)
with the only addition being the attaching of a Vector of spike times
to it with
	vs.play(spikevec)

  ------ How this works ------

The idiom for getting a  Vector argument
in a model description is encapsulated
in the "play" procedure. There are potentially many VecStim instances
and so the Vector pointer must be stored in the space allocated for the
particular instance when "play" is called. The assigned variable
"space" gives us space for a double precision number, 64 bits, which
is sufficient to store an opaque pointer. The "element" procedure
uses this opaque pointer to make sure that the requested "index" element
is within the size of the vector and assigns the "etime" double
precision variable to the value of that element. Since index is defined
at the model description level it is a double precision variable as
well and must be treated as such in the VERBATIM block. An index value of
-1 means that no further events should be sent from this instance.
Fortunately, space for model data is cleared when it is first allocated.
So if play is
not called, the pointer will be 0 and the test in the element procedure
would turn off the VecStim by setting index to -1. Also, because the
existence of the first argument is checked in the "play" procedure, one
can turn off the VecStim with
	vs.play()
No checking is done if the stimvec is destroyed (when the reference count
for the underlying Vector becomes 0). Continued use of the VecStim instance
in this case would cause a memory error. So it is up to the user to
call vs.play() or to destroy the VecStim instance
before running another simulation.

The strategy of the INITIAL and NET_RECEIVE blocks is to send a
self event (with flag 1) to be delivered at the time specified by
the index of the Vector starting at index 0. When the self event
is delivered to the NET_RECEIVE block, it causes an immediate
input event on every NetCon which has this VecStim as its source.
These events, would then be delivered to their targets after the
appropriate delay specified for each NetCon.

Currently, external events are ignored. 
It may be useful to elaborate the VecStim model in further analogy to
NetStim by allowing external events (remember, they have flag = 0)
to turn the VecStim on or off depending on the sign of the weight, w.
In this case, one would have to define the semantics of the spike
vector times as being relative to a start time and
introduce that "start" variable into the model. Presumabably, a negative
start time would mean that the VecStim is off.

ENDCOMMENT


:  Vector stream of events

NEURON {
	POINT_PROCESS VecStim
}

ASSIGNED {
	index
	etime (ms)
	space
}

INITIAL {
	index = 0
	element()
	if (index > 0) { 
		net_send(etime - t, 1)
	}
}

NET_RECEIVE (w) {
	if (flag == 1) {
		net_event(t)
		element()
		if (index > 0) {
                        if (etime < t) {
                            printf("Warning in VecStim: inter-spike interval smaller than dt.\nRounding up to dt.")
                            etime = t
                        }
			net_send(etime - t, 1)
		}
	}
}


VERBATIM
extern double* vector_vec();
extern int vector_capacity();
extern void* vector_arg();
ENDVERBATIM     

PROCEDURE element() {
VERBATIM
  { void* vv; int i, size; double* px;
	i = (int)index;
	if (i >= 0) {
		vv = *((void**)(&space));
		if (vv) {
			size = vector_capacity(vv);
			px = vector_vec(vv);
			if (i < size) {
				etime = px[i];
				index += 1.;
			}else{
				index = -1.;
			}
		}else{
			index = -1.;
		}
	}
  }
ENDVERBATIM
}

PROCEDURE play() {
VERBATIM
	void** vv;
	vv = (void**)(&space);
	*vv = (void*)0;
	if (ifarg(1)) {
		*vv = vector_arg(1);
	}
ENDVERBATIM
}
