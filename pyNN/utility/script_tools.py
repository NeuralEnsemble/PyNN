"""
A collection of functions to help writing simualtion scripts.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from datetime import datetime
import os
import sys
from importlib import import_module
import logging

# If there is a settings.py file on the path, defaults will be
# taken from there.
try:
    from settings import SMTPHOST, EMAIL
except ImportError:
    SMTPHOST = None
    EMAIL = None


def get_script_args(n_args, usage=''):
    """
    Get command line arguments.

    This works by finding the name of the main script and assuming any
    arguments after this in sys.argv are arguments to the script.
    It would be nicer to use optparse, but this doesn't seem to work too well
    with nrniv or mpirun.
    """
    calling_frame = sys._getframe(1)
    if '__file__' in calling_frame.f_locals:
        script = calling_frame.f_locals['__file__']
        try:
            script_index = sys.argv.index(script)
        except ValueError:
            try:
                script_index = sys.argv.index(os.path.abspath(script))
            except ValueError:
                script_index = 0
    else:
        script_index = 0
    args = sys.argv[script_index + 1:script_index + 1 + n_args]
    if len(args) != n_args:
        usage = usage or "Script requires %d arguments, you supplied %d" % (n_args, len(args))
        raise Exception(usage)
    return args


def get_simulator(*arguments):
    """
    Import and return a PyNN simulator backend module based on command-line
    arguments.

    The simulator name should be the first positional argument. If your script
    needs additional arguments, you can specify them as (name, help_text) tuples.
    If you need more complex argument handling, you should use argparse
    directly.

    Returns (simulator, command-line arguments)
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("simulator",
                        help="neuron, nest, brian or another backend simulator")
    for argument in arguments:
        arg_name, help_text = argument[:2]
        extra_args = {}
        if len(argument) > 2:
            extra_args = argument[2]
        parser.add_argument(arg_name, help=help_text, **extra_args)
    args = parser.parse_args()
    sim = import_module("pyNN.%s" % args.simulator)
    return sim, args


def init_logging(logfile, debug=False, num_processes=1, rank=0, level=None):
    """
    Simple configuration of logging.
    """
    # allow logfile == None
    # which implies output to stderr
    # num_processes and rank should be obtained using mpi4py, rather than having them as arguments
    if logfile:
        if num_processes > 1:
            logfile += '.%d' % rank
        logfile = os.path.abspath(logfile)

    # prefix log messages with mpi rank
    mpi_prefix = ""
    if num_processes > 1:
        mpi_prefix = 'Rank %d of %d: ' % (rank, num_processes)

    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # allow user to override exact log_level
    if level:
        log_level = level

    logging.basicConfig(
        level=log_level,
        format=mpi_prefix + '%(asctime)s %(levelname)-8s [%(name)s] %(message)s (%(pathname)s[%(lineno)d]:%(funcName)s)',  # noqa: E501
        filename=logfile,
        filemode='w')
    return logging.getLogger("PyNN")


def normalized_filename(root, basename, extension, simulator,
                        num_processes=None, use_iso8601=False):
    """
    Generate a file path containing a timestamp and information about the
    simulator used and the number of MPI processes.

    The date is used as a sub-directory name, the date & time are included in the
    filename.
    If use_iso8601 is True, follow https://en.wikipedia.org/wiki/ISO_8601
    """
    timestamp = datetime.now()
    if use_iso8601:
        date = timestamp.strftime("%Y-%m-%d")
        date_time = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        date = timestamp.strftime("%Y%m%d")
        date_time = timestamp.strftime("%Y%m%d-%H%M%S")

    if num_processes:
        np = "_np%d" % num_processes
    else:
        np = ""
    return os.path.join(root,
                        date,
                        "%s_%s%s_%s.%s" % (basename,
                                           simulator,
                                           np,
                                           date_time,
                                           extension))


def notify(
    msg="Simulation finished.",
    subject="Simulation finished.",
    smtphost=SMTPHOST,
    address=EMAIL,
):
    """Send an e-mail stating that the simulation has finished."""
    if not (smtphost and address):
        print(
            "SMTP host and/or e-mail address not specified.\n"
            "Unable to send notification message."
        )
    else:
        import smtplib

        msg = ("From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n") % (
            address,
            address,
            subject,
        ) + msg
        msg += "\nTimestamp: %s" % datetime.now().strftime("%H:%M:%S, %F")
        server = smtplib.SMTP(smtphost)
        server.sendmail(address, address, msg)
        server.quit()
