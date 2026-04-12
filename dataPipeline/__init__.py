"""Data processing pipeline for keystroke synthesis.

This package handles loading raw keystroke CSV/text file pairs,
cleaning and transforming the data, and preparing it for model training.

Modules:
    dataLoader: PyTorch Dataset that pairs text files with keystroke CSVs.
    dataPrepper: Cleans, filters, and feature-engineers raw keystroke CSV data.
"""

from .dataLoader import dataLoader
from .dataPrepper import dataPrepper

__all__ = ['dataLoader', 'dataPrepper']