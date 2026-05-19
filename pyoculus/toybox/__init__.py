"""
This submodule contains implementations of toy problems, datasets or functions. 
"""

import logging
logger = logging.getLogger(__name__)

try:
    from .cylindrical_toybox import *
except ImportError as e:
    logger.debug(str(e))