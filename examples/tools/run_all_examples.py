#!/usr/bin/env python

import subprocess
import glob
import os
import sys

default_simulators = ['MOCK', 'NEST', 'NEURON', 'Brian2']
simulator_names = sys.argv[1:]
if len(simulator_names) > 0:
    for name in simulator_names:
        if name not in default_simulators:
            print("Simulator must be one of: " + ", ".join(default_simulators))
            sys.exit(1)
else:
    simulator_names = default_simulators

simulators = []
for simulator in simulator_names:
    sim = simulator.lower()
    try:
        exec("import pyNN.%s" % sim)
        simulators.append(sim)
    except ImportError:
        pass

exclude = {
    'MOCK': ["nineml_neuron.py", "nineml_brunel.py"],
    'NEURON': ["nineml_neuron.py"],
    'NEST': ["nineml_neuron.py", "nineml_brunel.py"],
    'Brian2': ["nineml_neuron.py", "nineml_brunel.py", "multiquantal_synapses.py", "random_numbers.py",
               "gif_neuron.py", "stochastic_tsodyksmarkram.py", "stochastic_deterministic_comparison.py",
               "stochastic_synapses.py", "varying_poisson.py"]
}

extra_args = {
    "VAbenchmarks.py": "CUBA",
    "VAbenchmarks2.py": "CUBA",
    "VAbenchmarks2-csa.py": "CUBA",
    "VAbenchmarks3.py": "CUBA",
    "nineml_brunel.py": "SR"
}

if not os.path.exists("Results"):
    os.mkdir("Results")

for simulator in simulator_names:
    if simulator.lower() in simulators:
        print("\n\n\n================== Running examples with %s =================\n" % simulator)
        for script in glob.glob("../*.py"):
            script_name = os.path.basename(script)
            if script_name not in exclude[simulator]:
                cmd = "%s %s %s" % (sys.executable, script, simulator.lower())
                if script_name in extra_args:
                    cmd += " " + extra_args[script_name]
                print(cmd, end='')
                sys.stdout.flush()
                logfile = open("Results/%s_%s.log" % (os.path.basename(script), simulator), 'w')
                p = subprocess.Popen(cmd, shell=True, stdout=logfile,
                                     stderr=subprocess.PIPE, close_fds=True)
                retval = p.wait()
                print(retval == 0 and " - ok" or " - fail")
    else:
        print("\n\n\n================== %s not available =================\n" % simulator)

print("\n\n\n================== Plotting results =================\n")
for script in glob.glob("../*.py"):
    cmd = "%s plot_results.py %s" % (sys.executable, os.path.basename(script)[:-3])
    print(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, close_fds=True)
    p.wait()
