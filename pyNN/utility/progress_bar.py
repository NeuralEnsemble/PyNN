"""
Classes for showing progress bars in the shell during simulations.

:copyright: Copyright 2006-2024 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import sys


class ProgressBar(object):
    """
    Create a progress bar in the shell.
    """

    def __init__(self, width=77, char="#", mode="fixed"):
        self.char = char
        self.mode = mode
        if self.mode not in ['fixed', 'dynamic']:
            self.mode = 'fixed'
        self.width = width

    def set_level(self, level):
        """
        Rebuild the bar string based on `level`, which should be a number
        between 0 and 1.
        """
        if level < 0:
            level = 0
        if level > 1:
            level = 1

        # figure the proper number of 'character' make up the bar
        all_full = self.width - 2
        num_hashes = int(round(level * all_full))

        if self.mode == 'dynamic':
            # build a progress bar with self.char (to create a dynamic bar
            # where the percent string moves along with the bar progress.
            bar = self.char * num_hashes
        else:
            # build a progress bar with self.char and spaces (to create a
            # fixed bar (the percent string doesn't move)
            bar = self.char * num_hashes + ' ' * (all_full - num_hashes)
        bar = u'[ %s ] %3.0f%%' % (bar, 100 * level)
        print(bar, end=u' \r')
        sys.stdout.flush()

    def __call__(self, level):
        self.set_level(level)


class SimulationProgressBar(ProgressBar):

    def __init__(self, interval, t_stop, char="#", mode="fixed"):
        super().__init__(width=int(t_stop / interval), char=char, mode=mode)
        self.interval = interval
        self.t_stop = t_stop

    def __call__(self, t):
        self.set_level(t / self.t_stop)
        return t + self.interval
