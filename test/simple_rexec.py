"""
Very simple API for remotely executing jobs

Also see:
  http://jerith.za.net/code/remoteexec.html
  http://pussh.sourceforge.net
"""

from subprocess import Popen, PIPE
import sys
import os
from itertools import cycle


class Job(Popen):
    
    def __init__(self, script, *args):
        self.script = script
        self.args = args
        
    def run(self, node, output=None):
        self.node = node
        cmd = "ssh %s %s %s %s" % (node,
                                   sys.executable,
                                   os.path.abspath(self.script),
                                   " ".join(self.args))
        if output is None:
            output = PIPE
        Popen.__init__(self, cmd, stdin=None, stdout=output, stderr=output,
                       shell=True)

class JobManager(object):
    
    def __init__(self, node_list):
        self.node_list = node_list
        self._node = cycle(self.node_list)
        self.job_list = []
    
    def __iter__(self):
        return iter(self.job_list)
    
    def run(self, script, *args):
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