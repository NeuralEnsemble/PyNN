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
