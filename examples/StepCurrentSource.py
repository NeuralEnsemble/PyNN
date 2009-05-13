"""
Simple test of injecting time-varying current into a cell

Andrew Davison, UNIC, CNRS
May 2009

$Id:$
"""

import sys

simulator_name = sys.argv[-1]
exec("from pyNN.%s import *" % simulator_name)

setup()

cell = create(IF_curr_exp, {})
current_source = StepCurrentSource([50.0, 110.0, 150.0, 210.0],
                                   [0.1, 0.3, -0.2, 0.2])
cell.inject(current_source)

record_v(cell, "Results/StepCurrentSource_%s.v" % simulator_name)
run(250.0)
  
end()
