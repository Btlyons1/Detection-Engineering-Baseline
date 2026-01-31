"""
Detection Baseline Helper Functions
====================================

Reusable statistical and analysis functions for detection engineering baselines.
Designed to be imported into baseline notebooks to ensure consistency and
reduce code duplication across detection development.

Author: Security Engineering Team
Version: 1.1.0
License: MIT

Usage:
    from baseline_helpers import (
        load_data_from_duckdb,
        calculate_robust_statistics,
        detect_outliers_mad,
        FrequencyAnalyzer,
        BaselineDocumenter
    )
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Union, Tuple, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import sqlite3
from pathlib import Path

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data_from_duckdb(
    db_path: str,
    query: str,
    params: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Load data from DuckDB database.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database file
    query : str
        SQL query to execute
    params : tuple, optional
        Parameters for parameterized queries

    Returns
    -------
    pd.DataFrame
        Query results as DataFrame

    Example
    -------
    >>> df = load_data_from_duckdb(
    ...     "cloudtrail.duckdb",
    ...     "SELECT * FROM events WHERE event_date >= ?",
    ...     ("2024-01-01",)
    ... )
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("duckdb is required. Install with: pip install duckdb")

    conn = duckdb.connect(db_path, read_only=True)
    if params:
        df = conn.execute(query, params).fetchdf()
    else:
        df = conn.execute(query).fetchdf()
    conn.close()
    return df


def load_data_from_sqlite(
    db_path: str,
    query: str,
    params: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Load data from SQLite database (legacy support).

    Parameters
    ----------
    db_path : str
        Path to SQLite database file
    query : str
        SQL query to execute
    params : tuple, optional
        Parameters for parameterized queries

    Returns
    -------
    pd.DataFrame
        Query results as DataFrame

    Example
    -------
    >>> df = load_data_from_sqlite(
    ...     "cloudtrail.db",
    ...     "SELECT * FROM events WHERE event_date >= ?",
    ...     ("2024-01-01",)
    ... )
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_data_from_json(json_path: str) -> pd.DataFrame:
    """
    Load data from JSON file (pattern works for S3, data lakes, etc.)
    
    Parameters
    ----------
    json_path : str
        Path to JSON file
        
    Returns
    -------
    pd.DataFrame
        JSON data as DataFrame
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


# =============================================================================
# ROBUST STATISTICAL FUNCTIONS
# =============================================================================

def calculate_median(data: Union[pd.Series, np.ndarray]) -> float:
    """Calculate median - preferred over mean for security data."""
    return float(np.median(data))


def calculate_mad(data: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Median Absolute Deviation (MAD).
    
    MAD is a robust measure of dispersion that is not affected by outliers,
    making it ideal for security telemetry with long-tail distributions.
    
    MAD = median(|X_i - median(X)|)
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
        
    Returns
    -------
    float
        Median Absolute Deviation
    """
    median = np.median(data)
    return float(np.median(np.abs(data - median)))


def calculate_modified_zscore(
    data: Union[pd.Series, np.ndarray],
    consistency_constant: float = 0.6745
) -> np.ndarray:
    """
    Calculate Modified Z-Score using MAD.
    
    The modified z-score is robust to outliers and preferred for security data:
    
    Modified Z-Score = 0.6745 * (x_i - median) / MAD
    
    The constant 0.6745 makes MAD consistent with standard deviation for 
    normally distributed data.
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
    consistency_constant : float
        Constant for consistency with normal distribution (default: 0.6745)
        
    Returns
    -------
    np.ndarray
        Modified z-scores for each data point
        
    Reference
    ---------
    Iglewicz, B. and Hoaglin, D.C. (1993). How to Detect and Handle Outliers
    """
    data = np.asarray(data)
    median = np.median(data)
    mad = calculate_mad(data)
    
    # Handle case where MAD is 0 (all values are identical)
    if mad == 0:
        return np.zeros_like(data, dtype=float)
    
    return consistency_constant * (data - median) / mad


def detect_outliers_mad(
    data: Union[pd.Series, np.ndarray],
    threshold: float = 3.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Modified Z-Score method.
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
    threshold : float
        Modified z-score threshold for outlier detection (default: 3.5)
        Commonly used thresholds:
        - 2.5: More sensitive, catches borderline anomalies
        - 3.5: Standard, good balance (recommended)
        - 5.0: Conservative, only extreme outliers
        
    Returns
    -------
    tuple
        (outlier_mask, modified_z_scores)
        
    Example
    -------
    >>> outliers, scores = detect_outliers_mad(df['event_count'], threshold=3.5)
    >>> anomalous_records = df[outliers]
    """
    modified_z = calculate_modified_zscore(data)
    outlier_mask = np.abs(modified_z) > threshold
    return outlier_mask, modified_z


def calculate_percentiles(
    data: Union[pd.Series, np.ndarray],
    percentiles: List[float] = [5, 25, 50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """
    Calculate multiple percentiles for distribution analysis.
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
    percentiles : list
        Percentile values to calculate
        
    Returns
    -------
    dict
        Percentile values keyed by percentile name
    """
    return {f"p{p}": float(np.percentile(data, p)) for p in percentiles}


def calculate_iqr(data: Union[pd.Series, np.ndarray]) -> Tuple[float, float, float]:
    """
    Calculate Interquartile Range (IQR) bounds.
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound, iqr)
        Lower/upper bounds are Q1 - 1.5*IQR and Q3 + 1.5*IQR
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return float(lower_bound), float(upper_bound), float(iqr)


def calculate_robust_statistics(data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate comprehensive robust statistics for a dataset.
    
    Returns statistics that are resistant to outliers, making them
    suitable for security telemetry analysis.
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
        
    Returns
    -------
    dict
        Dictionary containing:
        - count: Number of observations
        - median: Central tendency (robust)
        - mad: Median Absolute Deviation
        - mean: Arithmetic mean (for reference, not recommended for thresholds)
        - std: Standard deviation (for reference)
        - iqr_lower: Lower IQR bound
        - iqr_upper: Upper IQR bound
        - percentiles: P5 through P99
        - skewness: Distribution skewness
        - kurtosis: Distribution kurtosis
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    iqr_lower, iqr_upper, iqr = calculate_iqr(data)
    percentiles = calculate_percentiles(data)
    
    return {
        "count": len(data),
        "median": float(np.median(data)),
        "mad": calculate_mad(data),
        "mean": float(np.mean(data)),  # For reference only
        "std": float(np.std(data)),    # For reference only
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "iqr": iqr,
        "iqr_lower_bound": iqr_lower,
        "iqr_upper_bound": iqr_upper,
        **percentiles,
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data))
    }


# =============================================================================
# FREQUENCY ANALYSIS CLASS
# =============================================================================

class FrequencyAnalyzer:
    """
    Analyze frequency patterns in categorical security data.
    
    Provides methods to understand the distribution of events, identify
    common vs rare patterns, and detect anomalous frequencies.
    
    Example
    -------
    >>> analyzer = FrequencyAnalyzer(df, 'event_name')
    >>> head, tail = analyzer.get_head_tail_analysis(head_pct=80)
    >>> rare_events = analyzer.get_rare_events(threshold=10)
    """
    
    def __init__(self, data: pd.DataFrame, column: str):
        """
        Initialize frequency analyzer.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data
        column : str
            Column name to analyze
        """
        self.data = data
        self.column = column
        self._frequency_table = None
        self._calculate_frequencies()
    
    def _calculate_frequencies(self):
        """Calculate and cache frequency table."""
        counts = self.data[self.column].value_counts()
        total = counts.sum()
        
        self._frequency_table = pd.DataFrame({
            'value': counts.index,
            'count': counts.values,
            'percentage': (counts.values / total * 100),
            'cumulative_percentage': (counts.values / total * 100).cumsum()
        })
    
    @property
    def frequency_table(self) -> pd.DataFrame:
        """Get the full frequency table."""
        return self._frequency_table.copy()
    
    def get_head_tail_analysis(
        self, 
        head_pct: float = 80
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into head (common) and tail (rare) based on cumulative percentage.
        
        This is critical for understanding long-tail distributions common in 
        security telemetry - often 20% of event types generate 80% of volume.
        
        Parameters
        ----------
        head_pct : float
            Cumulative percentage threshold for head (default: 80)
            
        Returns
        -------
        tuple
            (head_df, tail_df) - DataFrames for common and rare values
        """
        ft = self._frequency_table
        head = ft[ft['cumulative_percentage'] <= head_pct]
        tail = ft[ft['cumulative_percentage'] > head_pct]
        return head, tail
    
    def get_rare_events(self, threshold: int = 10) -> pd.DataFrame:
        """
        Get events occurring less than threshold times.
        
        Parameters
        ----------
        threshold : int
            Count threshold for "rare" events
            
        Returns
        -------
        pd.DataFrame
            Rare events below threshold
        """
        return self._frequency_table[self._frequency_table['count'] < threshold]
    
    def get_concentration_metrics(self) -> Dict[str, float]:
        """
        Calculate concentration metrics for the distribution.
        
        Returns metrics that help understand how concentrated or spread
        the distribution is.
        
        Returns
        -------
        dict
            - unique_count: Number of unique values
            - top_1_pct: Percentage of data from top 1 value
            - top_5_pct: Percentage of data from top 5 values
            - top_10_pct: Percentage of data from top 10 values
            - gini_coefficient: Measure of inequality (0=equal, 1=concentrated)
        """
        ft = self._frequency_table
        
        # Calculate Gini coefficient
        counts = ft['count'].values
        n = len(counts)
        if n == 0:
            gini = 0
        else:
            sorted_counts = np.sort(counts)
            cumulative = np.cumsum(sorted_counts)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return {
            'unique_count': len(ft),
            'top_1_pct': float(ft.iloc[0]['percentage']) if len(ft) > 0 else 0,
            'top_5_pct': float(ft.head(5)['percentage'].sum()) if len(ft) >= 5 else float(ft['percentage'].sum()),
            'top_10_pct': float(ft.head(10)['percentage'].sum()) if len(ft) >= 10 else float(ft['percentage'].sum()),
            'gini_coefficient': float(gini)
        }


# =============================================================================
# BASELINE DOCUMENTATION CLASS
# =============================================================================

@dataclass
class DetectionBaseline:
    """
    Data class for storing detection baseline results.
    
    Captures all relevant information for documentation and future reference.
    """
    # Metadata
    detection_name: str
    detection_id: str
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    version: str = "1.0.0"
    
    # Hypothesis
    hypothesis: str = ""
    background: str = ""
    
    # Data scope
    data_source: str = ""
    date_range_start: str = ""
    date_range_end: str = ""
    total_records: int = 0
    
    # Statistical findings
    frequency_analysis: Dict = field(default_factory=dict)
    distribution_stats: Dict = field(default_factory=dict)
    grouping_analysis: Dict = field(default_factory=dict)
    
    # Thresholds
    recommended_thresholds: Dict = field(default_factory=dict)
    
    # Findings
    findings: List[str] = field(default_factory=list)
    anomalies_detected: List[Dict] = field(default_factory=list)
    
    # Validation
    validation_results: Dict = field(default_factory=dict)


class BaselineDocumenter:
    """
    Document and persist baseline analysis results.
    
    Provides methods to save baseline findings for:
    - Historical reference
    - LLM analysis
    - Scheduled job comparison
    - Rolling baseline updates
    
    Example
    -------
    >>> documenter = BaselineDocumenter("./baselines")
    >>> baseline = DetectionBaseline(
    ...     detection_name="Unusual API Call Volume",
    ...     detection_id="DET-001"
    ... )
    >>> baseline.distribution_stats = stats
    >>> documenter.save_baseline(baseline)
    """
    
    def __init__(self, output_dir: str = "./baseline_outputs"):
        """
        Initialize documenter with output directory.
        
        Parameters
        ----------
        output_dir : str
            Directory for storing baseline outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_baseline(
        self, 
        baseline: DetectionBaseline,
        format: str = "json"
    ) -> str:
        """
        Save baseline to file.
        
        Parameters
        ----------
        baseline : DetectionBaseline
            Baseline object to save
        format : str
            Output format ('json' or 'markdown')
            
        Returns
        -------
        str
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{baseline.detection_id}_{timestamp}"
        
        if format == "json":
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(asdict(baseline), f, indent=2, default=str)
        elif format == "markdown":
            filepath = self.output_dir / f"{filename}.md"
            with open(filepath, 'w') as f:
                f.write(self._to_markdown(baseline))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def _to_markdown(self, baseline: DetectionBaseline) -> str:
        """Convert baseline to markdown format."""
        md = f"""# Detection Baseline: {baseline.detection_name}

**Detection ID:** {baseline.detection_id}  
**Version:** {baseline.version}  
**Created:** {baseline.created_date}  
**Author:** {baseline.author}  

## Background

{baseline.background}

## Hypothesis

{baseline.hypothesis}

## Data Scope

- **Source:** {baseline.data_source}
- **Date Range:** {baseline.date_range_start} to {baseline.date_range_end}
- **Total Records:** {baseline.total_records:,}

## Statistical Analysis

### Distribution Statistics

```json
{json.dumps(baseline.distribution_stats, indent=2, default=str)}
```

### Frequency Analysis

```json
{json.dumps(baseline.frequency_analysis, indent=2, default=str)}
```

### Grouping Analysis

```json
{json.dumps(baseline.grouping_analysis, indent=2, default=str)}
```

## Recommended Thresholds

```json
{json.dumps(baseline.recommended_thresholds, indent=2, default=str)}
```

## Findings

"""
        for i, finding in enumerate(baseline.findings, 1):
            md += f"{i}. {finding}\n"
        
        md += f"""
## Anomalies Detected

{json.dumps(baseline.anomalies_detected, indent=2, default=str)}

## Validation Results

```json
{json.dumps(baseline.validation_results, indent=2, default=str)}
```
"""
        return md
    
    def load_baseline(self, filepath: str) -> DetectionBaseline:
        """
        Load a previously saved baseline.
        
        Parameters
        ----------
        filepath : str
            Path to baseline JSON file
            
        Returns
        -------
        DetectionBaseline
            Loaded baseline object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return DetectionBaseline(**data)
    
    def compare_baselines(
        self, 
        current: DetectionBaseline, 
        previous: DetectionBaseline
    ) -> Dict[str, Any]:
        """
        Compare two baselines to detect drift.
        
        Parameters
        ----------
        current : DetectionBaseline
            Current baseline
        previous : DetectionBaseline
            Previous baseline for comparison
            
        Returns
        -------
        dict
            Comparison results including drift metrics
        """
        comparison = {
            'detection_id': current.detection_id,
            'current_date': current.created_date,
            'previous_date': previous.created_date,
            'record_count_change': current.total_records - previous.total_records,
            'record_count_change_pct': (
                (current.total_records - previous.total_records) / previous.total_records * 100
                if previous.total_records > 0 else 0
            ),
            'stat_changes': {}
        }
        
        # Compare key statistics if available
        for stat_key in ['median', 'mad', 'p95', 'p99']:
            if (stat_key in current.distribution_stats and 
                stat_key in previous.distribution_stats):
                curr_val = current.distribution_stats[stat_key]
                prev_val = previous.distribution_stats[stat_key]
                comparison['stat_changes'][stat_key] = {
                    'current': curr_val,
                    'previous': prev_val,
                    'change': curr_val - prev_val,
                    'change_pct': (
                        (curr_val - prev_val) / prev_val * 100
                        if prev_val != 0 else 0
                    )
                }
        
        return comparison


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_baseline_coverage(
    df: pd.DataFrame,
    group_column: str,
    expected_groups: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate that baseline covers expected groups/entities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Baseline data
    group_column : str
        Column to check for coverage
    expected_groups : list, optional
        Expected values that should be present
        
    Returns
    -------
    dict
        Coverage validation results
    """
    observed = set(df[group_column].unique())
    
    result = {
        'total_groups_observed': len(observed),
        'observed_groups': list(observed)
    }
    
    if expected_groups:
        expected = set(expected_groups)
        result['expected_groups'] = list(expected)
        result['missing_groups'] = list(expected - observed)
        result['unexpected_groups'] = list(observed - expected)
        result['coverage_pct'] = len(expected & observed) / len(expected) * 100
    
    return result


def suggest_threshold_from_stats(
    stats: Dict[str, float],
    method: str = 'mad',
    sensitivity: str = 'medium'
) -> Dict[str, float]:
    """
    Suggest detection thresholds based on statistical analysis.
    
    Parameters
    ----------
    stats : dict
        Output from calculate_robust_statistics()
    method : str
        Threshold method ('mad', 'percentile', 'iqr')
    sensitivity : str
        Detection sensitivity ('low', 'medium', 'high')
        
    Returns
    -------
    dict
        Suggested thresholds with methodology notes
    """
    sensitivity_multipliers = {
        'high': {'mad': 2.5, 'iqr': 1.0},    # More sensitive
        'medium': {'mad': 3.5, 'iqr': 1.5},  # Balanced
        'low': {'mad': 5.0, 'iqr': 2.0}      # Conservative
    }
    
    sensitivity_percentiles = {
        'high': 90,
        'medium': 95,
        'low': 99
    }
    
    multipliers = sensitivity_multipliers[sensitivity]
    percentile = sensitivity_percentiles[sensitivity]
    
    result = {
        'method': method,
        'sensitivity': sensitivity
    }
    
    if method == 'mad':
        threshold = stats['median'] + (multipliers['mad'] * stats['mad'] / 0.6745)
        result['threshold'] = threshold
        result['formula'] = f"median + ({multipliers['mad']} * MAD / 0.6745)"
        result['modified_z_threshold'] = multipliers['mad']
        
    elif method == 'percentile':
        result['threshold'] = stats[f'p{percentile}']
        result['formula'] = f"P{percentile}"
        
    elif method == 'iqr':
        result['threshold'] = stats['iqr_upper_bound']
        result['formula'] = f"Q3 + {multipliers['iqr']} * IQR"
    
    return result
