"""
Script to run a test many times, with parameters taken from a ParameterSpace
object. If run on a cluster, the runs will be distributed across different nodes.

For a full description of usage, run:

python explore_space.py --help

"""

import os
import tempfile
from optparse import OptionParser
import socket
from NeuroTools.parameters import ParameterSpace
from NeuroTools import datastore
from simple_rexec import JobManager


def read_hostfile(option, opt, value, parser):
    f = open(os.path.expanduser(value), 'r')
    parser.values.host_list = f.read().split()
    f.close()

def parse_hostlist(option, opt, value, parser):
    parser.values.host_list = value.split(',')

# read command-line arguments
usage = "Usage: %prog [options] test_script parameter_file [script_args]"
parser = OptionParser(usage)
parser.add_option("-n", "--trials", type=int, dest="trials", default=1,
                  help="Number of values to draw from each parameter distribution")
parser.add_option("-H", "--hosts", action="callback", callback=parse_hostlist, type=str,
                  help="Comma-separated list of hosts to run jobs on")
parser.add_option("-f", "--hostfile", action="callback", callback=read_hostfile,
                  type=str, help="Provide a hostfile")
(options, args) = parser.parse_args()
if len(args) < 2:
    parser.error("incorrect number of arguments")

test_script, url = args[:2]
script_args = args[2:]
trials = options.trials
if hasattr(options, "host_list"):
    host_list = options.host_list
else:
    host_list = [socket.gethostname()] # by default, run just on the current host

# iterate over the parameter space, creating a job each time
parameter_space = ParameterSpace(url)
tempfiles = []
job_manager = JobManager(host_list, delay=0, quiet=False)

for sub_parameter_space in parameter_space.iter_inner(copy=True):
    for parameter_set in sub_parameter_space.realize_dists(n=trials, copy=True):
        ##print parameter_set.pretty()
        fd, tmp_url = tempfile.mkstemp(dir=os.getcwd())
        os.close(fd)
        tempfiles.append(tmp_url)
        parameter_set.save(tmp_url)
        job_manager.run(test_script, parameter_set._url, *script_args)

# wait until all jobs have finished    
job_manager.wait()

# retrieve results stored by the jobs. We use the NeuroTools datastore, so
# the test module itself is the storage key.
test_module = __import__(test_script.replace(".py",""))
ds = datastore.ShelveDataStore(root_dir=parameter_space.results_dir,
                               key_generator=datastore.keygenerators.hash_pickle)
#ds = datastore.DjangoORMDataStore(database_parameters={'DATABASE_ENGINE': 'sqlite3',
#                                                       'DATABASE_NAME': '%s/datastore.db' % parameter_space.results_dir},
#                                  data_root_dir=parameter_space.results_dir)

for job in job_manager:
    ##print job.read_output()
    test_module.parameters = test_module.load_parameters(job.args[0])
    print ds.retrieve(test_module, 'distances')
    print ds.retrieve(test_module, 'vm_diff')
    ds.store(test_module, 'output', job.output)

# clean up
for file in tempfiles:
    os.remove(file)
    