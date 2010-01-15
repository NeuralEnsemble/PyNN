from pyNN.neuroml import *

setup(file="test.xml")

p1 = Population((5,5), IF_cond_alpha, {'tau_m': 10.0})
p2 = Population((2,3,2), IF_cond_exp, {'tau_m': 15.0})

#prj = Projection(p1, p2, 'fixedProbability', 0.5)

end() 