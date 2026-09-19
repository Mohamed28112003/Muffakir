"""
Results module for Muffakir Composer.

Contains data classes for storing trial results and aggregated reports.
"""

from .trial import TrialResult
from .report import ComposerReport

__all__ = ["TrialResult", "ComposerReport"]