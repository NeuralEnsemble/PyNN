"""
A collection of utility functions.
$Id:$
"""

# If there is a settings.py file in the PyNN root directory, defaults will be
# taken from there.
try:
    from pyNN.settings import SMTPHOST, EMAIL
except ImportError:
    SMTPHOST = None
    EMAIL = None
import sys

red     = 0010; green  = 0020; yellow = 0030; blue = 0040;
magenta = 0050; cyan   = 0060; bright = 0100
try:
    import ll.ansistyle
    def colour(col, text):
        return str(ll.ansistyle.Text(col,str(text)))
except ImportError:
    def colour(col, text):
            return text


def notify(msg="Simulation finished.", subject="Simulation finished.", smtphost=SMTPHOST, address=EMAIL):
        """Send an e-mail stating that the simulation has finished."""
        if not (smtphost and address):
            print "SMTP host and/or e-mail address not specified.\nUnable to send notification message."
        else:
            import smtplib, datetime
            msg = ("From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n") % (address,address,subject) + msg
            msg += "\nTimestamp: %s" % datetime.datetime.now().strftime("%H:%M:%S, %F")
            server = smtplib.SMTP(smtphost)
            server.sendmail(address, address, msg)
            server.quit()

def get_script_args(script, n_args):
    script_index = sys.argv.index(script)
    args = sys.argv[script_index+1:script_index+1+n_args]
    if len(args) != n_args:
        raise Exception("Script requires %d arguments, you supplied %d" % (n_args, len(args)))
    return args
    
