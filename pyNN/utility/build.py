"""
Tools for building simulator extensions

"""

import os
import subprocess


def run_command(path, working_directory):
    p = subprocess.Popen(path, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True,
                         close_fds=True, cwd=working_directory)
    result = p.wait()
    stdout = p.stdout.readlines()
    p.stdout.close()
    return result, stdout


def find(command):
    """Try to find an executable file."""
    path = os.environ.get("PATH", "").split(os.pathsep)
    cmd = ''
    for dir_name in path:
        abs_name = os.path.abspath(os.path.normpath(os.path.join(dir_name, command)))
        if os.path.isfile(abs_name):
            cmd = abs_name
            break
    return cmd


def compile_nmodl(mechanism_directory):
    nrnivmodl = find("nrnivmodl")
    if nrnivmodl:
        print("nrnivmodl found at", nrnivmodl)
        result, stdout = run_command(nrnivmodl, mechanism_directory)
        # test if nrnivmodl was successful
        if result != 0:
            print("Unable to compile NEURON extensions. Output was:")
            print('  '.join([''] + stdout))  # indent error msg for easy comprehension
        else:
            print("Successfully compiled NEURON extensions.")
    else:
        print("Unable to find nrnivmodl. It will not be possible to use the pyNN.neuron module.")


def compile_nest_extensions(extension_directory, nest_build_dir="_build"):
    nest_config = find("nest-config")
    if nest_config:
        print("nest-config found at", nest_config)
        if not os.path.exists(nest_build_dir):
            os.mkdir(nest_build_dir)
        result, stdout = run_command("cmake -Dwith-nest={} {}".format(nest_config, extension_directory),
                                     nest_build_dir)
        if result != 0:
            print("Problem running cmake. Output was:")
            print('  '.join([''] + stdout))
        else:
            result, stdout = run_command("make", nest_build_dir)
            if result != 0:
                print("Unable to compile NEST extensions. Output was:")
                print('  '.join([''] + stdout))
            else:
                result, stdout = run_command("make install", nest_build_dir)  # should really move this to install stage
                if result != 0:
                    print("Unable to install NEST extensions. Output was:")
                    print('  '.join([''] + stdout))
                else:
                    print("Successfully compiled NEST extensions.")
    else:
        print("Unable to find nest-config. You can use NEST built-in models, but it will not be possible to use NEST extensions")
