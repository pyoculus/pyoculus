import logging
logger = logging.getLogger(__name__)

from .magnetic_field import MagneticField
from .cylindrical_grid_interpolated_field import AxisymmetricCylindricalGridField

# Cartesian magnetic fields
# from .cartesian_bfield import CartesianBfield

# Toroidal magnetic fields
from .toroidal_bfield import ToroidalBfield
from .two_waves import TwoWaves
# from .qfm_bfield import QFMBfield
from .spec_bfield import SpecBfield
from .spectre_bfield import SpectreBfield

# Cylindrical magnetic fields
from .cylindrical_bfield import CylindricalBfield
try:
    from .simsopt_bfield import SimsoptBfield
except ImportError as e:
    logger.debug(e)
try:
    from .cylindrical_bfield_analytic import AnalyticCylindricalBfield
except ImportError as e:
    logger.debug(e)

# from .m3dc1_bfield import M3DC1Bfield