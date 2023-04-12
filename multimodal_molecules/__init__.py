from dunamai import Version

try:
    version = Version.from_any_vcs()
    __version__ = version.serialize()
    del version
except RuntimeError:
    __version__ = None
