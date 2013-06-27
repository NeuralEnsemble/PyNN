: Izhikevich artificial neuron model from 
: EM Izhikevich "Simple Model of Spiking Neurons"
: IEEE Transactions On Neural Networks, Vol. 14, No. 6, November 2003 pp 1569-1572

NEURON {
  POINT_PROCESS Izhikevich
  RANGE a, b, c, d
  NONSPECIFIC_CURRENT i_inj
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (Gohm) = (gigaohm)
}

INITIAL {
  vm = -70
  u = -14
  net_send(0, 1)
}

CONSTANT {
    Rm = 1 (Gohm)
}

PARAMETER {
  a       = 0.02 (/ms)
  b       = 0.2  (/mV)
  c       = -65  (mV)   : reset potential after a spike
  d       = 2    (mV/ms)
  vthresh = 30   (mV)   : spike threshold
  i_inj   = 0    (nA)
}

STATE { 
  u (mV/ms)
  vm (mV)
}

BREAKPOINT {
  SOLVE states METHOD derivimplicit
  :printf("v=%f u=%f, dv=%f, du=%f\n", vm, u, 0.04*vm*vm + 5*vm + 140 - u + i_inj*Rm, a*(b*vm-u))
}

UNITSOFF

DERIVATIVE states {
  vm' = 0.04*vm*vm + 5*vm + 140 - u + i_inj*Rm
  u' = a*(b*vm-u) 
}

UNITSON

NET_RECEIVE (weight (mV)) {
  if (flag == 1) {
    WATCH (vm > vthresh) 2
  } else if (flag == 2) {
    net_event(t)
    vm = c
    u = u + d
  } else { : synaptic activation
    vm = vm + weight
  }
}
