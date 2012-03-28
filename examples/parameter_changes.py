"""

"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("import pyNN.%s as sim" % simulator_name)

sim.setup(timestep=0.01)

cell = sim.Population(1, sim.EIF_cond_exp_isfa_ista, {'v_thresh': -55.0, 'tau_refrac': 5.0})
current_source = sim.StepCurrentSource({'times' : [50.0, 200.0, 250.0, 400.0, 450.0, 600.0, 650.0, 800.0],
                                        'amplitudes' : [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]})
cell.inject(current_source)
cell.record_v()

for a in (0.0, 4.0, 20.0, 100.0):
    print "Setting current to %g nA" % a
    cell.set(a=a)
    sim.run(200.0)
  
cell.print_v("Results/parameter_changes_%s.v" % simulator_name)
  
sim.end()