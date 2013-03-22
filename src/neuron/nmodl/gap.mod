NEURON {
    POINT_PROCESS Gap
    RANGE g, i, vgap
    NONSPECIFIC_CURRENT i
}

UNITS {
  (nA) = (nanoamp)
  (mV) = (millivolt)
  (uS) = (nanosiemens)
}

PARAMETER { g = 0 (uS) }
    
ASSIGNED {
    v    (mV)
    vgap (mV)
    i    (nA)
}
 
BREAKPOINT { 
  i = g * (v-vgap)
}
