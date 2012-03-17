"""
Example script to help explore what needs to be done to integrate MUSIC with PyNN

This example corresponds to test/randevents.music in the MUSIC distribution

"""

from pyNN import music

ext1, ext2 = music.setup(music.Config(2, "eventgenerator", "-b 1 10"),
                         music.Config(2, "eventlogger", "-b 2"))

music.connect (ext1, "out", ext2, "in", width = 10)

music.run(100.0)
