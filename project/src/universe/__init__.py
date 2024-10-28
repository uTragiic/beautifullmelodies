"""
Universe management module for NYSE trading.
Provides functionality for stock filtering, clustering, and universe maintenance.
"""

from .manager import UniverseManager
from .clustering import UniverseClusterer, ClusterConfig
from .filters import UniverseFilter, FilterConfig

__all__ = [
    'UniverseManager',
    'UniverseClusterer',
    'ClusterConfig',
    'UniverseFilter',
    'FilterConfig'
]

# Version information
__version__ = '1.0.0'

# Module level docstring
"""
Universe management package for the trading system.

This package provides components for:
- Universe selection and filtering
- Stock clustering and grouping
- Data quality management
- Universe maintenance and updates

Main components:
- UniverseManager: Coordinates universe selection and updates
- UniverseClusterer: Handles stock clustering and grouping
- UniverseFilter: Implements filtering criteria
"""