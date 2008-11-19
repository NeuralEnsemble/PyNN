"""
Very simple API for remotely executing jobs

Also see:
  http://jerith.za.net/code/remoteexec.html
  http://pussh.sourceforge.net
  http://www.theether.org/pssh/
"""

from subprocess import Popen, PIPE, STDOUT
import sys
import os
from StringIO import StringIO
from itertools import cycle
import tempfile
from time import sleep

class Job(Popen):
    
    def __init__(self, script, *args):
        self.script = script
        self.args = args
        
    def run(self, node):
        self.node = node
        cmd = "ssh -x %s %s %s %s" % (node,
                                      sys.executable,
                                      os.path.abspath(self.script),
                                      " ".join(self.args))
        self._output = tempfile.TemporaryFile()
        Popen.__init__(self, cmd, stdin=None, stdout=self._output, stderr=STDOUT,
                       shell=True)

    def wait(self):
        Popen.wait(self)
        self._output.seek(0)
        self.output = self._output.read()
        self._output.close()

    def read_output(self):
        prefix = "\n%s " % self.node
        return prefix.join(self.output.split("\n"))


class JobManager(object):
    
    def __init__(self, node_list, delay=0):
        self.node_list = node_list
        self._node = cycle(self.node_list)
        self.job_list = []
        self.delay = delay
    
    def __iter__(self):
        return iter(self.job_list)
    
    def run(self, script, *args):
        """
        Run the script on the next node in the list.
        
        There is no attempt at load balancing. It might be better to define the
        maximal number of jobs for each node, and take the next node as the one
        with the smallest number of jobs (using poll() instead of wait()).
        """
        sleep(self.delay)
        job = Job(script, *args)
        node = self._node.next()
        job.run(node)
        self.job_list.append(job)
        
    def wait(self):
        for job in self.job_list:
            job.wait()
    
def test():
    if not os.path.exists("test.py"):
        f = open("test.py", 'w')
        f.write('import socket, time\ntime.sleep(1)\nprint socket.gethostname()\n')
        f.close()
    node_list = ['upstate', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8']
    jm = JobManager(node_list)
    for node in node_list:
        jm.run("test.py")
    jm.wait()
    for job in jm:
        print job.stdout.read()

if __name__ == "__main__":
    test()