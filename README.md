# Detection Engineering Baseline

A practical framework for building detection baselines using statistical methods. This repository accompanies the blog post on detection baselining techniques for security engineering.

> **Blog Post:** [Detection Engineering: Building Baselines That Actually Work](https://your-substack-url-here.substack.com/p/detection-baselines)

## Overview

Detection baselines help security teams distinguish normal behavior from anomalies by quantifying "what does normal look like?" using robust statistical methods. This repository demonstrates:

- **Robust Statistics**: Using Median and MAD (Median Absolute Deviation) instead of mean/std for skewed security data
- **Modified Z-Score**: Outlier detection that's resistant to extreme values
- **Frequency Analysis**: Understanding long-tail distributions common in security telemetry
- **Contextual Baselines**: Per-user and temporal analysis for better signal-to-noise

## Project Structure

```
Detection-Engineering-Baseline/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Reusable modules
│   ├── __init__.py
│   └── baseline_helpers.py   # Statistical functions, analyzers, documenters
├── scripts/                  # Utility scripts
│   └── generate_synthetic_cloudtrail.py  # Synthetic data generator
├── data/                     # Generated data files
│   ├── cloudtrail_baseline.duckdb        # DuckDB database
│   └── cloudtrail_events.json            # JSON backup
└── detections/               # Detection baselines
    └── DET-2026-001/         # Example: Unusual API Call Volume
        ├── detection_baseline_notebook.ipynb
        └── outputs/          # Generated visualizations & reports
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python scripts/generate_synthetic_cloudtrail.py
```

This creates a DuckDB database with ~24,000 synthetic CloudTrail events including three injected anomalies:
- **Day 15**: Off-hours credential creation (`iam:CreateAccessKey` at 3 AM)
- **Day 20**: Bulk S3 data access (500 `s3:GetObject` events)
- **Day 25**: Unknown external user `sts:AssumeRole` attempts

### 3. Run the Baseline Notebook

Open the Jupyter notebook and run all cells:

```bash
cd detections/DET-2026-001
jupyter notebook detection_baseline_notebook.ipynb
```

Or in VS Code, open the notebook and select "Run All Cells".

## Key Concepts

### Why Robust Statistics?

Security telemetry is typically **right-skewed with heavy tails**. A small number of extreme events (legitimate or malicious) can heavily distort mean and standard deviation:

| Metric | Value | Problem |
|--------|-------|---------|
| Mean | Inflated by outliers | Threshold too high, misses anomalies |
| Std Dev | Inflated by outliers | Wide "normal" range, low sensitivity |
| **Median** | Robust to outliers | Stable central tendency |
| **MAD** | Robust to outliers | Stable spread measure |

### Modified Z-Score

```
Modified Z = 0.6745 × (x - median) / MAD
```

The constant 0.6745 scales MAD to be comparable to standard deviation. Observations with |Modified Z| > 3.5 are flagged as outliers.

### The Baseline Workflow

1. **Frequency Analysis** - What happens most often? What's rare?
2. **Distribution Analysis** - Calculate robust statistics (median, MAD, percentiles)
3. **Grouping Analysis** - Per-user, per-hour, per-source patterns
4. **Outlier Detection** - Apply Modified Z-Score to flag anomalies
5. **Threshold Selection** - Choose detection threshold based on acceptable FP rate
6. **Validation** - Backtest against known anomalies
7. **Documentation** - Persist findings for reproducibility

## Module Reference

### `src/baseline_helpers.py`

#### Data Loading
```python
from src.baseline_helpers import load_data_from_duckdb

df = load_data_from_duckdb("data/cloudtrail_baseline.duckdb", "SELECT * FROM cloudtrail_events")
```

#### Statistical Functions
```python
from src.baseline_helpers import calculate_robust_statistics, detect_outliers_mad

stats = calculate_robust_statistics(df['event_count'])
# Returns: median, mad, mean, std, percentiles, skewness, kurtosis

outlier_mask, z_scores = detect_outliers_mad(df['daily_count'], threshold=3.5)
```

#### Frequency Analysis
```python
from src.baseline_helpers import FrequencyAnalyzer

analyzer = FrequencyAnalyzer(df, 'user_name')
freq_table = analyzer.frequency_table
concentration = analyzer.get_concentration_metrics()
head, tail = analyzer.get_head_tail_analysis(head_pct=80)
rare_events = analyzer.get_rare_events(threshold=10)
```

## Detection Naming Convention

Detections follow the format: `DET-YYYY-NNN`

- `DET` - Detection prefix
- `YYYY` - Year created
- `NNN` - Sequential number

Each detection gets its own directory under `detections/` containing:
- Jupyter notebook with analysis
- `outputs/` directory for visualizations and reports

## Outputs

The notebook generates 10 visualizations saved to `detections/DET-2026-001/outputs/`:

| File | Description |
|------|-------------|
| `user_activity_distribution.png` | User activity bar chart with Pareto analysis |
| `api_frequency.png` | Top 15 most common API calls |
| `temporal_distribution.png` | Hourly event distribution (business vs off-hours) |
| `distribution_analysis.png` | Histogram and box plots of daily volumes |
| `outlier_detection.png` | Time series with outliers highlighted |
| `per_user_baselines.png` | Per-user median vs P95 comparison |
| `off_hours_analysis.png` | Off-hours ratio analysis with heatmap |
| `ip_distribution.png` | Source IP distribution (internal vs external) |
| `threshold_comparison.png` | Comparison of threshold methods |
| `summary_dashboard.png` | Executive summary visualization |

## Adapting for Production

The patterns demonstrated here are portable to production data platforms:

| This Example | Production Alternative |
|--------------|----------------------|
| DuckDB | Snowflake, Databricks, BigQuery, Athena |
| Local notebook | Scheduled job, Airflow DAG |
| JSON output | SIEM integration, alert API |
| Synthetic data | Real CloudTrail, Okta, etc. |

## License

MIT

## Author

Brandon Lyons 

---

*For questions or feedback, please open an issue or reach out via the blog post comments.*
