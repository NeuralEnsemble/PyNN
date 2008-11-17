
import sys
import os
from NeuroTools.parameters import ParameterSpace
from subprocess import Popen, PIPE
import tempfile
from simple_rexec import JobManager

#node_list = ['upstate', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']
node_list = ['node3', 'node4']
job_manager = JobManager(node_list)

test_script = sys.argv[1]
url = sys.argv[2]
trials = int(sys.argv[3])
parameter_space = ParameterSpace(url)

tempfiles = []

for parameter_set in parameter_space.realize_dists(n=trials, copy=True):
    print parameter_set.pretty()
    fd, tmp_url = tempfile.mkstemp(dir=os.getcwd())
    os.close(fd)
    tempfiles.append(tmp_url)
    parameter_set.save(tmp_url)
    job_manager.run(test_script, parameter_set._url)
    
job_manager.wait()

for job in job_manager:
    print job.node.join(job.stdout.xreadlines())
    
for file in tempfiles:
    os.remove(file)
    