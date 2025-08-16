"""
sldata: A Python library for neural spike data analysis.

This package provides tools for loading and analyzing preprocessed neural spike data
and behavioral event data.

Main Classes:
- SessionData: Core class for loading and analyzing neural session data

Utilities:
- behavior_utils: Utilities for behavioral data processing and TTL handling

Code by Nate Gonzales-Hess, August 2025.
"""

from .SessionData import SessionData
from . import behavior_utils

__version__ = "0.1.0"
__author__ = "Nate Gonzales-Hess"
__email__ = "nate.gonzales.hess@gmail.com"

# Make main classes and modules available at package level
__all__ = [
    "SessionData",
    "behavior_utils",
]