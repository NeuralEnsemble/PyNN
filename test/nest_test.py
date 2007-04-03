from pyNN.nest import *

cell_params = {'Tau': 20.0,
               'C'  : 1.0}

E_net = Population((5,5),"iaf_neuron",cellparams=cell_params,label="E_net")
E_net.set('Tau',30)

pynest.end()