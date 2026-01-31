"""
Detection Engineering Baseline - Source Module
===============================================

This module provides reusable functions and classes for building
detection engineering baselines using statistical methods.

Modules:
    baseline_helpers: Statistical analysis, outlier detection, and documentation tools
"""

from .baseline_helpers import (
    # Data Loading
    load_data_from_duckdb,
    load_data_from_sqlite,
    load_data_from_json,

    # Statistical Functions
    calculate_robust_statistics,
    detect_outliers_mad,
    calculate_modified_zscore,
    calculate_percentiles,
    calculate_iqr,
    calculate_mad,

    # Analysis Classes
    FrequencyAnalyzer,
    DetectionBaseline,

    # Documentation
    BaselineDocumenter,

    # Utilities
    suggest_threshold_from_stats,
    validate_baseline_coverage,
)

__version__ = "1.1.0"
__author__ = "Security Engineering Team"
