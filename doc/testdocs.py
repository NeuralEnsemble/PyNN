#!/usr/bin/env python
"""
Script to run doctests.
"""

import doctest
import sys
import os
from optparse import OptionParser

optionflags = doctest.IGNORE_EXCEPTION_DETAIL + doctest.NORMALIZE_WHITESPACE
optionflags = doctest.NORMALIZE_WHITESPACE


class MyOutputChecker(doctest.OutputChecker):
    """
    Modification of doctest.OutputChecker to work better with the PyNN
    users' manual:
      * Often, we don't want to have the output that is printed
    by Python in the manual, as it just takes up space without adding any
    useful information.
    """

    def __init__(self, strict):
        self.strict = strict

    def check_output(self, want, got, optionflags):
        if self.strict:
            return doctest.OutputChecker.check_output(self, want, got, optionflags)
        else:
            if want == '':
                return True
            else:
                try:
                    long(want) and long(got)  # where the output is an id
                    return True
                except ValueError:
                    try:
                        if round(float(want), 8) == round(float(got), 8):
                            return True
                        else:
                            return doctest.OutputChecker.check_output(self, want, got, optionflags)
                    except ValueError:
                        return doctest.OutputChecker.check_output(self, want, got, optionflags)


def mytestfile(filename, globs, optionflags, strict=False):
    parser = doctest.DocTestParser()
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    name = os.path.basename(filename)

    runner = doctest.DocTestRunner(checker=MyOutputChecker(strict=strict), optionflags=optionflags)
    # Read the file, convert it to a test, and run it.
    s = open(filename).read()
    test = parser.get_doctest(s, globs, name, filename, 0)
    runner.run(test)
    runner.summarize()
    return runner.failures, runner.tries


def print_script(filename, simulator):
    parser = doctest.DocTestParser()
    s = open(filename).read()
    script = "".join([ex.source for ex in parser.get_examples(s) if "+SKIP" not in ex.source])
    print("from pyNN.%s import *\nsetup(max_delay=10.0, debug=True)\n%s" % (simulator, script))


def remove_data_files():
    import glob
    for pattern in ("*.dat", "*.npz", "*.h5", "*.conn", "logfile"):
        for filename in glob.glob(pattern):
            os.remove(filename)


# ==============================================================================
if __name__ == "__main__":

    # Process command line
    parser = OptionParser(usage="usage: %prog [options] FILE")
    parser.add_option("-s", "--simulator", dest="simulator",
                      type="choice", choices=('nest', 'neuron', 'brian'),
                      help="run doctests with SIMULATOR", metavar="SIMULATOR",
                      default='nest')
    parser.add_option("--strict", action="store_true", dest="strict", default=False,
                      help="Use the original doctest output checker, not the more lax local version.")
    parser.add_option("-p", "--print", action="store_true", default=False, dest="dump",
                      help="Just print out the script extracted from the document, don't run the test.")

    if 'nrniv' in sys.argv[0]:
        (options, args) = parser.parse_args(sys.argv[5:])
    else:
        (options, args) = parser.parse_args()
    if len(args) == 1:
        docfile = args[0]
    else:
        parser.print_help()
        sys.exit(1)

    # Run test
    if options.dump:
        print_script(docfile, options.simulator)
    else:
        exec("from pyNN.%s import *" % options.simulator)
        setup(max_delay=10.0, debug=True)
        if options.simulator == "neuron":
            create(IF_curr_alpha)  # this is to use up ID 0, making the IDs agree with NEST.
        mytestfile(docfile, globs=globals(), optionflags=optionflags, strict=options.strict)

    remove_data_files()
    sys.exit(0)
