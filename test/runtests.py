#!/usr/bin/env python
""" Master script for running tests.
$Id$
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
    elif 'neuron' in engine:
        cmd = '../hoc/i686/special -python ' + cmd + '.py %s' % engine
    else:
        print 'Invalid simulation engine "%s". Valid values are "nest", "oldneuron" and "neuron"' % engine
        
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

def sortTracesByCells(traces, gids):
    # First, we see what are the gid present in the recorded file:
    cells_id = []
    for gid in gids:
        if not gid in cells_id:
            cells_id.append(gid)
    # Then we return a list of tuples containing the Vm traces sorted
    # by the cell_ids:
    cells_id = N.sort(cells_id)
    result = N.array([])
    for id in cells_id:
        idx = N.where(gids == id)
        result = N.concatenate((result, traces[idx]))
    return result


def compare_traces(script,mse_threshold,engines):
    """For scripts that write a voltage trace to file."""
    
    print script, ": ",
    traces = {}
    fail = False
    
    for engine in engines:
        traces[engine] = []
        run(script, engine)
        filenames = glob.glob('%s_*_%s.v' % (script, engine))
        if len(filenames) == 0:
            filenames = glob.glob('%s_%s.v' % (script, engine))
        if filenames:
            for filename in filenames:
                f = open(filename,'r')
                trace = [line.strip() for line in f.readlines() if line[0] != "#"]
                f.close()
                trace = [line for line in trace if line] # ignore blank lines
                try:
                    if engine == 'oldneuron':
                        position = N.zeros(len(trace))
                    else:
                        position = N.array([int(line.split()[1]) for line in trace]) # take second column
                except IndexError:
                    print engine
                    print line
                    raise
                trace = N.array([float(line.split()[0]) for line in trace]) # take first column 
                trace = sortTracesByCells(trace, position)
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
    
    engine_list = ("nest", "oldneuron", "neuron")
    
    thresholds = {"IF_curr_alpha" : 0.5,
                  "IF_curr_alpha2" : 5.0,
                  "IF_curr_exp" : 0.1, 
                  "IF_cond_alpha" : 0.5,
                  "simpleNetworkL" : 0.5,
                  "simpleNetwork" : 0.6,
                  "small_network" : 6.0}
    if len(sys.argv) > 1:
        scripts = sys.argv[1:]
    else:
        scripts = thresholds.keys()
    for script in scripts:
        compare_traces(script, thresholds[script]*(len(engine_list)-1), engine_list)
    
