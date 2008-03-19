"""
Test of the EIF_cond_alpha_isfa_ista model

Andrew Davison, UNIC, CNRS
December 2007

$Id:$
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "oldneuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.01,min_delay=0.01,max_delay=4.0,debug=True)

ifcell = create(EIF_cond_alpha_isfa_ista,
                {'i_offset': 1.0, 'tau_refrac': 2.0, 'v_spike': -40})
print ifcell.getParameters()
if simulator == "neuron":
    hoc.execute("cell0.cell psection()")
    
record_v(ifcell,"EIF_cond_alpha_isfa_ista.py_%s.v" % simulator)
run(200.0)

end()

