"""
Test of the EIF_cond_alpha_isfa_ista model

Andrew Davison, UNIC, CNRS
December 2007

"""

from pyNN.utility import get_script_args, normalized_filename

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.01,min_delay=0.1,max_delay=4.0,debug=True)

ifcell = create(EIF_cond_alpha_isfa_ista(i_offset=1.0, tau_refrac=2.0, v_spike=-40))
print ifcell[0].get_parameters()

filename = normalized_filename("Results", "EIF_cond_alpha_isfa_ista", "pkl",
                               simulator_name)
record('v', ifcell, filename, annotations={'script_name': __file__})
run(200.0)

end()
