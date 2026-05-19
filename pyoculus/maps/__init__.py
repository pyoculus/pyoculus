"""
This submodule provides the maps.
"""


from .base_map import BaseMap
from .integrated_map import IntegratedMap
import logging
logger = logging.getLogger(__name__)

# from .standard_map import StandardMap

from .toroidal_bfield_section import ToroidalBfieldSection
from .cylindrical_bfield_section import CylindricalBfieldSection
try: 
    from .tokamap import TokaMap
except ImportError as e:
    logger.debug(e)
try: 
    from .standard_map import StandardMap
except ImportError as e:
    logger.debug(e)
# from .spec_pjh import *
