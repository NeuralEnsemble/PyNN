"""


"""

import os
import tempfile
import shutil
import subprocess
from itertools import cycle
from datetime import datetime

simulators = cycle(["nest", "neuron"])

examples = (
    # "HH_cond_exp2.py",
    "Izhikevich.py",
    "current_injection.py",
    # "VAbenchmarks.py",
    # "brunel.py",
    "cell_type_demonstration.py",
    # "connections.py",
    # "distrib_example.py",
    # "inhomogeneous_network.py",
    # "nineml_neuron.py",
    # "parameter_changes.py",
    "random_numbers.py",
    "random_distributions.py",
    # "simpleRandomNetwork.py",
    # "simpleRandomNetwork_csa.py",
    "simple_STDP.py",
    "small_network.py",
    # "specific_network.py",
    # "stdp_network.py",
    "synaptic_input.py",
    "tsodyksmarkram.py",
    "varying_poisson.py",
    "stochastic_synapses.py",
    "stochastic_deterministic_comparison.py"
)

# todo: add line numbering to code examples

template = """{title}
{underline}

.. image:: ../images/examples/{img_file}

.. literalinclude:: ../../examples/{example}

"""

example_index = """========
Examples
========

.. toctree::
   :maxdepth: 2

"""

image_dir = "images/examples"
examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "examples")
tmp_dir = tempfile.mkdtemp()
results_dir = os.path.join(tmp_dir, "Results")

for dir_name in (image_dir, results_dir):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def run(python_script, simulator, *extra_args):
    files_initial = list_files(".png")
    args = " ".join(extra_args)
    p = subprocess.Popen("python %s/%s --plot-figure %s %s" % (examples_dir, python_script, simulator, args),
                         shell=True, cwd=tmp_dir)
    p.wait()
    new_files = list_files(".png").difference(files_initial)
    return new_files


def get_title(python_script):
    with open(os.path.join(examples_dir, python_script), "r") as fp:
        while True:
            line = fp.readline()
            if line[:3] == '"""':
                title = fp.readline().strip().strip(".")
                break
    return title


def list_files(filter):
    return set([os.path.join(x[0], filename)
                for x in os.walk(results_dir)
                for filename in x[2]
                if filter in filename])


print("Running examples in {}".format(tmp_dir))
for example in examples:
    new_files = run(example, next(simulators))
    if len(new_files) > 1:
        raise Exception("Multiple image files")
    img_path, = new_files
    shutil.copy(img_path, image_dir)
    img_file = os.path.basename(img_path)
    title = get_title(example)
    underline = "=" * len(title)
    with open(os.path.join("examples", example.replace(".py", ".txt")), "w") as fp:
        fp.write(template.format(**locals()))
    example_index += "   examples/{}\n".format(example.replace(".py", ""))

# handle VAbenchmarks separately
example = "VAbenchmarks.py"
cell_type = "CUBA"
files_initial = list_files("VAbenchmarks")
run(example, "nest", cell_type)
run(example, "neuron", cell_type)
new_files = list_files("VAbenchmarks").difference(files_initial)
files_initial = list_files(".png")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
p = subprocess.Popen("python %s/tools/VAbenchmark_graphs.py -o Results/VAbenchmarks_%s_%s.png %s" % (
    examples_dir, cell_type, timestamp, " ".join(new_files)),
    shell=True, cwd=tmp_dir)
p.wait()
img_path, = list_files(".png").difference(files_initial)
shutil.copy(img_path, image_dir)
img_file = os.path.basename(img_path)
title = get_title(example)
underline = "=" * len(title)
with open(os.path.join("examples", example.replace(".py", ".txt")), "w") as fp:
    fp.write(template.format(**locals()))
example_index += "   examples/{}\n".format(example.replace(".py", ""))

with open("examples.txt", "w") as fp:
    fp.write(example_index)

shutil.rmtree(tmp_dir)
