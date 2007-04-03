#!/usr/bin/env python
""" Master script for running tests.
$Id: runtests.py 14 2007-01-30 13:09:03Z apdavison $
"""

import subprocess, sys, glob
import numpy as N

nrnheader = """NEURON -- Version 5.8 2005-10-14 12:36:20 Main (88)
by John W. Moore, Michael Hines, and Ted Carnevale
Duke and Yale University -- Copyright 1984-2005

Additional mechanisms from files
 alphaisyn.mod alphasyn.mod expisyn.mod refrac.mod reset.mod stdwa_softlimits.mod stdwa_songabbott.mod stdwa_symm.mod vecstim.mod"""

def run(cmd,engine):
    #print 'Running "', cmd, '" with', engine.upper()
    logfile = open("%s_%s.log" % (cmd,engine), 'w')
    if engine == 'nest':
        cmd = 'python ' + cmd + '.py nest'
    elif engine[:6] == 'neuron':
        cmd = '../hoc/i686/special -python ' + cmd + '.py neuron'
    else:
        print 'Invalid simulation engine "%s". Valid values are "nest" and "neuron"' % engine
        
    p = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=subprocess.PIPE, close_fds=True)
    p.wait()
    logfile.close()
    errorMsg = p.stderr.read()
    errorMsg = errorMsg.strip(nrnheader)
    
    if len(errorMsg) > 0:
        print "=== %s Error =======================" % engine.upper()
        print "  " + errorMsg.replace("\n","\n   ")
        print "========================================"
        sys.exit(2)

def compare_traces(script,mse_threshold,engines):
    """For scripts that write a voltage trace to file."""
    
    print script, ": ",
    traces = {}
    fail = False
    
    for engine in engines:
        traces[engine] = []
        run(script, engine)
        filenames = glob.glob('%s_*%s*.v' % (script, engine))
        if filenames:
            for filename in filenames:
                f = open(filename,'r')
                trace = [line.strip() for line in f.readlines() if line[0] != "#"]
                f.close()
                trace = N.array([float(line.split()[0])  for line in trace if line]) # only take first column and ignore blank lines
                traces[engine].append(trace)
        else:
            fail = True; fail_message = "No files match glob pattern"

    mse = 0.0
    engine1 = engines[0] # compare all against the first engine in the list.
    for engine2 in engines[1:]:
        for trace1,trace2 in zip(traces[engine1],traces[engine2]):
            l1 = len(trace1); l2 = len(trace2)
            if l1 > 0 and l2 > 0 :
                if l1 > l2:
                    trace1 = trace1[:l2]
                elif l2 > l1:
                    trace2 = trace2[:l1]
                
                diff = trace1 - trace2
                mse += N.sqrt(N.mean(N.square(diff)))
            else:
                fail = True; fail_message = "empty file"
    
    if not fail:
        if mse < mse_threshold:
            print "Pass (mse = %f, threshold = %f)" % (mse, mse_threshold)
        else:
            print "Fail (mse = %f, threshold = %f)" % (mse, mse_threshold)
    else:
        print "Fail - ", fail_message
    
    

if __name__ == "__main__":
    
    engine_list = ("nest", "neuron")
    
    thresholds = {"IF_curr_alpha"  : 0.5,
                  "IF_curr_alpha2" : 5.0,
                  "IF_curr_exp"    : 0.1,
                  "simpleNetworkL" : 0.5,
                  "simpleNetwork"  : 0.5,
                  "small_network"  : 5.0}
    if len(sys.argv) > 1:
        scripts = sys.argv[1:]
    else:
        scripts = thresholds.keys()
    for script in scripts:
        compare_traces(script, thresholds[script]*(len(engine_list)-1), engine_list)
    