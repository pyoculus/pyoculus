try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata

__version__ = metadata.version('pyoculus')

from . import fields
from . import maps
from . import integrators
from . import utils
from . import solvers
from . import geo