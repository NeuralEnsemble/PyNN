#!/usr/bin/env python
""" Master script for running tests.
$Id$
"""

import subprocess, sys, glob, os
import numpy as N

nrnheader = """NEURON -- Version 5.8 2005-10-14 12:36:20 Main (88)
by John W. Moore, Michael Hines, and Ted Carnevale
Duke and Yale University -- Copyright 1984-2005

Additional mechanisms from files
 alphaisyn.mod alphasyn.mod expisyn.mod refrac.mod reset.mod stdwa_softlimits.mod stdwa_songabbott.mod stdwa_symm.mod vecstim.mod"""

def run(cmd,engine):
    #print 'Running "', cmd, '" with', engine.upper()
    logfile = open("%s_%s.log" % (cmd,engine), 'w')
    if engine in ('nest1', 'pcsim', 'nest2'):
        cmd = 'python ' + cmd + '.py ' + engine
    elif 'neuron' in engine:
        cmd = '../hoc/i686/special -python ' + cmd + '.py %s' % engine
    else:
        print 'Invalid simulation engine "%s". Valid values are "nest1", "nest2", "pcsim", "oldneuron" and "neuron"' % engine
        
    p = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=subprocess.PIPE, close_fds=True)
    p.wait()
    logfile.close()
    errorMsg = p.stderr.read()
    errorMsg = errorMsg.strip(nrnheader)
    
    if len(errorMsg) > 0:
        print "\n=== %s Error =======================" % engine.upper()
        print "  " + errorMsg.replace("\n","\n   ")
        print "======================================="
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
    fail_message = ""
    
    for engine in engines:
        traces[engine] = []
        try:
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
                fail = True; fail_message += "No files match glob pattern. "
        except Exception:
            fail = True
            fail_message += "Exception raised in %s. " % engine.upper()

    if not fail:
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
                    fail = True;
                    if (l1 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine1.upper(), script)
                    if (l2 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine2.upper(), script)
    if not fail:
        if mse < mse_threshold:
            print " Pass (mse = %f, threshold = %f)" % (mse, mse_threshold)
        else:
            print " Fail (mse = %f, threshold = %f)" % (mse, mse_threshold)
    else:
        print "Fail - ", fail_message


def compare_rasters(script,mse_threshold,engines):
    """For scripts that write a voltage trace to file."""
    
    print script, ": ",
    rasters = {}
    fail = False
    fail_message = ""
    
    for engine in engines:
        rasters[engine] = []
        #try:
        if (True):
            run(script, engine)
            filenames = glob.glob('%s_*_%s.ras' % (script, engine))
            if len(filenames) == 0:
                filenames = glob.glob('%s_%s.ras' % (script, engine))
            if filenames:
                for filename in filenames:
                    f = open(filename,'r')
                    raster = [line.strip() for line in f.readlines() if line[0] != "#"]
                    f.close()
                    raster = [line for line in raster if line] # ignore blank lines
                    try:
                        if engine == 'oldneuron':
                            position = N.zeros(len(raster))
                        else:
                            position = N.array([int(line.split()[1]) for line in raster]) # take second column
                    except IndexError:
                        print engine
                        print line
                        raise
                    raster = N.array([float(line.split()[0]) for line in raster]) # take first column 
                    raster = sortTracesByCells(raster, position)
                    rasters[engine].append(raster)
            else:
                fail = True; fail_message += "No files match glob pattern. "
        #except Exception:
        #    fail = True
        #    fail_message += "Exception raised in %s. " % engine.upper()

    if not fail:
        mse  = 0.0
        engine1 = engines[0] # compare all against the first engine in the list.
        for engine2 in engines[1:]:
            for raster1,raster2 in zip(rasters[engine1],rasters[engine2]):
                l1 = len(raster1); l2 = len(raster2)
                if l1 > 0 and l2 > 0 :
                    diff = []
                    for idx in xrange(len(raster1)):
                        diff.append((raster1[idx]-raster2).min())
                    mse += N.sqrt(N.mean(N.square(N.array(diff))))
                else:
                    fail = True;
                    if (l1 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine1.upper(), script)
                    if (l2 == 0):
                        fail_message = "%s produce an empty file for %s" %(engine2.upper(), script)
    if not fail:
        if mse < mse_threshold:
            print " Pass (mse = %f, threshold = %f)" % (mse, mse_threshold)
        else:
            print " Fail (mse = %f, threshold = %f)" % (mse, mse_threshold)
    else:
        print "Fail - ", fail_message
    

if __name__ == "__main__":
    
    engine_list = ("nest1", "oldneuron", "neuron", "pcsim", "nest2")
    
    thresholds_v = {"IF_curr_alpha"  : 0.26,
                    "IF_curr_exp"    : 0.25, 
                    "IF_cond_alpha"  : 0.25,
                    "IF_cond_exp"    : 0.25,
                    "simpleNetworkL" : 0.5,
                    "simpleNetwork"  : 0.7,
                    "IF_curr_alpha2" : 5.0,
                    "small_network"  : 5.0}
    
    thresholds_ras = {"SpikeSourcePoisson" : 50.}
    
    scripts_v   = []
    scripts_ras = []
    
    if len(sys.argv) > 1:
        for item in sys.argv[1:]:
            if item in thresholds_v.keys():
                scripts_v.append(item)
            if item in thresholds_ras.keys():
                scripts_ras.append(item)
    else:
        scripts_v   = thresholds_v.keys()
        scripts_ras = thresholds_ras.keys()

    for script in scripts_v:
        compare_traces(script, thresholds_v[script]*(len(engine_list)-1), engine_list)

    for script in scripts_ras:
        compare_rasters(script, thresholds_ras[script]*(len(engine_list)-1), engine_list)
    
