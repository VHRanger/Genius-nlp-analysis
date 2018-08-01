# GeniusAPI
# John W. Miller
# See LICENSE for details

"""A library that provides a Python interface to the Genius API"""

__author__ = 'John W. Miller'
__url__ = 'https://github.com/johnwmillr/GeniusAPI'
__description__ = 'A Python wrapper around the Genius API'
__license__ = 'MIT'
__version__ = '0.7.0'

import sys
assert sys.version_info[0] == 3, "LyricsGenius requires Python 3."
from .api import Genius
