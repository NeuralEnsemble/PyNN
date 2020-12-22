###################################################
###     	Simulation parameters		###
###################################################

simulator_params = {
    'nest':
    {
        'timestep': 0.1,    # ms
        'threads': 1,
        'sim_duration': 1000.,  # ms
    }
}

system_params = {
    # number of MPI nodes
    'n_nodes': 1,
    # number of MPI processes per node
    'n_procs_per_node': 2,
    # walltime for simulation
    'walltime': '8:0:0',
    # total memory for simulation
    'memory': '4gb',

    # file name for standard output
    'outfile': 'output.txt',
    # file name for error output
    'errfile': 'errors.txt',
    # absolute path to which the output files should be written
    'output_path': 'results',
    # path to the MPI shell script
    'mpi_path': '',
    # path to back-end
    'backend_path': '',
    # path to pyNN installation
    'pyNN_path': '',
    # command for submitting the job
    'submit_cmd': 'qsub'
}
