"""timefence: a point-in-time correct credit feature store."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("timefence")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0+unknown"
