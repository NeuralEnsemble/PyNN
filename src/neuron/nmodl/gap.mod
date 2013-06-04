NEURON {

    POINT_PROCESS Gap
    RANGE g, i, vremote
    NONSPECIFIC_CURRENT i
}

UNITS {

  (nA) = (nanoamp)
  (mV) = (millivolt)
  (nS) = (nanosiemens)
}

PARAMETER { g = 0 (uS) }
    
ASSIGNED {

    v    (mV)
    vremote (mV)
    i    (nA)
}
 
BREAKPOINT { 

  if (g > 0) { i = g * (v - vremote) }

}
