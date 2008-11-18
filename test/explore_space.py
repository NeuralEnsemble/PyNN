"""
Script to run a test many times, with parameters taken from a ParameterSpace
object. If run on a cluster, the runs will be distributed across different nodes.

Usage:

python explore_space.py <test_script> <parameter_file> <trials>

"""

import sys
import os
from NeuroTools.parameters import ParameterSpace
from NeuroTools import datastore
from subprocess import Popen, PIPE
import tempfile
from simple_rexec import JobManager

# node_list should really be read from file given on command line
node_list = ['upstate', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']

# read command-line arguments
test_script = sys.argv[1]
url = sys.argv[2]
trials = int(sys.argv[3])

# iterate over the parameter space, creating a job each time
parameter_space = ParameterSpace(url)
tempfiles = []
job_manager = JobManager(node_list)

for parameter_set in parameter_space.realize_dists(n=trials, copy=True):
    ##print parameter_set.pretty()
    fd, tmp_url = tempfile.mkstemp(dir=os.getcwd())
    os.close(fd)
    tempfiles.append(tmp_url)
    parameter_set.save(tmp_url)
    job_manager.run(test_script, parameter_set._url)

# wait until all jobs have finished    
job_manager.wait()

# retrieve results stored by the jobs. We use the NeuroTools datastore, so
# the test module itself is the storage key.
test_module = __import__(test_script.replace(".py",""))
ds = datastore.ShelveDataStore(root_dir=parameter_space.results_dir,
                               key_generator=datastore.keygenerators.hash_pickle)

for job in job_manager:
    print job.read_output()
    test_module.parameters = test_module.load_parameters(job.args[0])
    print ds.retrieve(test_module, 'distances')
    ds.store(test_module, 'output', job.output)

# clean up
for file in tempfiles:
    os.remove(file)
    