"""
A Timer class for timing script execution.

:copyright: Copyright 2006-2024 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import time
from ..core import deprecated


class Timer(object):
    """
    For timing script execution.

    Timing starts on creation of the timer.
    """

    def __init__(self):
        self.start()
        self.marks = []

    def start(self):
        """Start/restart timing."""
        self._start_time = time.perf_counter()
        self._last_check = self._start_time

    def elapsed_time(self, format=None):
        """
        Return the elapsed time in seconds but keep the clock running.

        If called with ``format="long"``, return a text representation of the
        time. Examples::

            >>> timer.elapsed_time()
            987
            >>> timer.elapsed_time(format='long')
            16 minutes, 27 seconds
        """
        current_time = time.perf_counter()
        elapsed_time = current_time - self._start_time
        if format == "long":
            elapsed_time = Timer.time_in_words(elapsed_time)
        self._last_check = current_time
        return elapsed_time

    @deprecated("elapsed_time()")
    def elapsedTime(self, format=None):
        return self.elapsed_time(format)

    def reset(self):
        """Reset the time to zero, and start the clock."""
        self.start()

    def diff(
        self, format=None
    ):  # I think delta() would be a better name for this method.
        """
        Return the time since the last time :meth:`elapsed_time()` or
        :meth:`diff()` was called.

        If called with ``format='long'``, return a text representation of the
        time.
        """
        current_time = time.perf_counter()
        time_since_last_check = current_time - self._last_check
        self._last_check = current_time
        if format == "long":
            time_since_last_check = Timer.time_in_words(time_since_last_check)
        return time_since_last_check

    @staticmethod
    def time_in_words(s):
        """
        Formats a time in seconds as a string containing the time in days,
        hours, minutes, seconds. Examples::

            >>> Timer.time_in_words(1)
            1 second
            >>> Timer.time_in_words(123)
            2 minutes, 3 seconds
            >>> Timer.time_in_words(24*3600)
            1 day
        """
        # based on http://mail.python.org/pipermail/python-list/2003-January/181442.html
        T = {}
        T["year"], s = divmod(s, 31556952)
        min, T["second"] = divmod(s, 60)
        h, T["minute"] = divmod(min, 60)
        T["day"], T["hour"] = divmod(h, 24)

        def add_units(val, units):
            return "%d %s" % (int(val), units) + (val > 1 and "s" or "")

        return ", ".join(
            [
                add_units(T[part], part)
                for part in ("year", "day", "hour", "minute", "second")
                if T[part] > 0
            ]
        )

    def mark(self, label):
        """
        Store the time since the last time since the last time
        :meth:`elapsed_time()`, :meth:`diff()` or :meth:`mark()` was called,
        together with the provided label, in the attribute 'marks'.
        """
        self.marks.append((label, self.diff()))
