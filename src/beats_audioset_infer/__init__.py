"""Lazy public BEATs AudioSet inference facade. Reads: session on demand."""
from .__about__ import __version__
__all__ = ['BEATsAudioSetSession', '__version__']
def __getattr__(name):
    if name == 'BEATsAudioSetSession':
        from .session import BEATsAudioSetSession
        return BEATsAudioSetSession
    raise AttributeError(name)
