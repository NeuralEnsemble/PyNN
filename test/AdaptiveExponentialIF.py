"""
Test of the AdaptiveExponentialIF_alpha/_exp models

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

ifcell = create(AdaptiveExponentialIF_alpha,
                {'i_offset': 1.0})
print ifcell.getParameters()
hoc.execute("cell0.cell psection()")
    
record_v(ifcell,"AdaptiveExponentialIF_%s.v" % simulator)
run(200.0)

end()

