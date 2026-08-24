"""Minimal public surface required by the dedicated TI2V worker.

The upstream package imports speech, animation and pose pipelines from its
top-level module. Those unrelated pipelines pull optional dependencies that
the TI2V-only worker neither uses nor should download.
"""

from . import configs, distributed, modules
from .textimage2video import WanTI2V

__all__ = ["WanTI2V"]
