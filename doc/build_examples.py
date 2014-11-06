"""


"""

import os
import tempfile
import shutil

import subprocess

examples = (
    # "HH_cond_exp2.py",
    "Izhikevich.py",
    "current_injection.py",
    # "VAbenchmarks.py",
    # "brunel.py",
    "cell_type_demonstration.py",
    #"connections.py",
    # "distrib_example.py",
    # "inhomogeneous_network.py",
    # "nineml_neuron.py",
    # "parameter_changes.py",
    # "random_distributions.py",
    "random_numbers.py",
    # "simpleRandomNetwork.py",
    # "simpleRandomNetwork_csa.py",
    "simple_STDP.py",
    "small_network.py",
    # "specific_network.py",
    # "stdp_network.py",
    "synaptic_input.py",
    # "tsodyksmarkram.py",
    # "varying_poisson.py",
)

# todo: add line numbering to code examples

template = """{title}
{underline}

.. literalinclude:: ../../examples/{example}

.. image:: ../images/examples/{img_file}

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

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def run(python_script):
    p = subprocess.Popen("python %s/%s --plot-figure nest" % (examples_dir, python_script), shell=True, cwd=tmp_dir)
    p.wait()


def get_title(python_script):
    with open(os.path.join(examples_dir, python_script), "r") as fp:
        while True:
            line = fp.readline()
            if line[:3] == '"""':
                title = fp.readline().strip().strip(".")
                break
    return title


files_initial = set([os.path.join(x[0], filename) for x in os.walk(results_dir) for filename in x[2] if ".png" in filename])
for example in examples:
    run(example)
    files = set([os.path.join(x[0], filename) for x in os.walk(results_dir) for filename in x[2] if ".png" in filename])
    new_files = files.difference(files_initial)
    files_initial = files
    if len(new_files) > 1:
        raise Exception("Multiple image files")
    img_path, = new_files
    shutil.copy(img_path, image_dir)
    img_file = os.path.basename(img_path)
    title = get_title(example)
    underline = "="*len(title)
    with open(os.path.join("examples", example.replace(".py", ".txt")), "w") as fp:
        fp.write(template.format(**locals()))
    example_index += "   examples/{}\n".format(example.replace(".py", ""))

with open("examples.txt", "w") as fp:
    fp.write(example_index)

shutil.rmtree(tmp_dir)
