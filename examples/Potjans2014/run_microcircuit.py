from sim_params import system_params
import os
import shutil

# Creates output folder if it does not exist yet, creates sim_script.sh,
# and submits it to the queue

system_params['num_mpi_procs'] = system_params['n_nodes'] * system_params['n_procs_per_node']

# Copy simulation scripts to output directory
try:
    os.mkdir(system_params['output_path'])
except OSError:
    pass

shutil.copy('network_params.py', system_params['output_path'])
shutil.copy('sim_params.py', system_params['output_path'])
shutil.copy('microcircuit.py', system_params['output_path'])
shutil.copy('network.py', system_params['output_path'])
shutil.copy('connectivity.py', system_params['output_path'])
shutil.copy('scaling.py', system_params['output_path'])
shutil.copy('plotting.py', system_params['output_path'])

job_script_template = """
#PBS -o %(output_path)s/%(outfile)s
#PBS -e %(output_path)s/%(errfile)s 
#PBS -l walltime=%(walltime)s
#PBS -l nodes=%(n_nodes)d:ppn=%(n_procs_per_node)d
#PBS -q intel
#PBS -l mem=%(memory)s
. %(mpi_path)s
mpirun -np %(num_mpi_procs)d python %(output_path)s/microcircuit.py
"""

f = open(system_params['output_path'] + '/sim_script.sh', 'w')
f.write(job_script_template % system_params)
f.close()

os.system('cd %(output_path)s && %(submit_cmd)s sim_script.sh' % system_params)
