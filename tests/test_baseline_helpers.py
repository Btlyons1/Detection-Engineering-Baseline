"""
Tests for baseline_helpers module.

Run with: pytest tests/test_baseline_helpers.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_helpers import (
    # Statistical functions
    calculate_median,
    calculate_mad,
    calculate_modified_zscore,
    detect_outliers_mad,
    calculate_percentiles,
    calculate_iqr,
    calculate_robust_statistics,
    # Analysis classes
    FrequencyAnalyzer,
    DetectionBaseline,
    BaselineDocumenter,
    # Utilities
    suggest_threshold_from_stats,
    validate_baseline_coverage,
    load_data_from_json,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Simple dataset for basic tests."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def skewed_data():
    """Right-skewed dataset typical of security telemetry."""
    return np.array([1, 1, 2, 2, 2, 3, 3, 4, 5, 100])


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for frequency analysis."""
    return pd.DataFrame({
        'user_name': ['alice', 'alice', 'alice', 'bob', 'bob', 'charlie'],
        'event_name': ['read', 'read', 'write', 'read', 'delete', 'read'],
        'event_count': [10, 20, 5, 15, 3, 8]
    })


# =============================================================================
# STATISTICAL FUNCTION TESTS
# =============================================================================

class TestCalculateMedian:
    """Tests for calculate_median function."""

    def test_odd_length(self):
        data = np.array([1, 2, 3, 4, 5])
        assert calculate_median(data) == 3.0

    def test_even_length(self):
        data = np.array([1, 2, 3, 4])
        assert calculate_median(data) == 2.5

    def test_with_pandas_series(self):
        data = pd.Series([1, 2, 3, 4, 5])
        assert calculate_median(data) == 3.0

    def test_single_value(self):
        data = np.array([42])
        assert calculate_median(data) == 42.0


class TestCalculateMAD:
    """Tests for calculate_mad function."""

    def test_basic_mad(self):
        # Data: [1, 2, 3, 4, 5], median = 3
        # Deviations from median: [2, 1, 0, 1, 2]
        # MAD = median([0, 1, 1, 2, 2]) = 1
        data = np.array([1, 2, 3, 4, 5])
        assert calculate_mad(data) == 1.0

    def test_constant_data(self):
        # All values same -> MAD = 0
        data = np.array([5, 5, 5, 5, 5])
        assert calculate_mad(data) == 0.0

    def test_with_outlier(self):
        # MAD should be resistant to outliers
        data_normal = np.array([1, 2, 3, 4, 5])
        data_with_outlier = np.array([1, 2, 3, 4, 1000])

        mad_normal = calculate_mad(data_normal)
        mad_outlier = calculate_mad(data_with_outlier)

        # MAD should not change dramatically with single outlier
        assert mad_outlier == mad_normal  # median of deviations unchanged


class TestCalculateModifiedZscore:
    """Tests for calculate_modified_zscore function."""

    def test_median_has_zero_zscore(self):
        data = np.array([1, 2, 3, 4, 5])
        z_scores = calculate_modified_zscore(data)
        # The median value (3) should have z-score of 0
        assert z_scores[2] == 0.0

    def test_symmetry(self):
        data = np.array([1, 2, 3, 4, 5])
        z_scores = calculate_modified_zscore(data)
        # Values equidistant from median should have opposite z-scores
        assert np.isclose(z_scores[0], -z_scores[4])
        assert np.isclose(z_scores[1], -z_scores[3])

    def test_constant_data_returns_zeros(self):
        data = np.array([5, 5, 5, 5, 5])
        z_scores = calculate_modified_zscore(data)
        assert np.all(z_scores == 0)

    def test_outlier_has_high_zscore(self, skewed_data):
        z_scores = calculate_modified_zscore(skewed_data)
        # The outlier (100) should have the highest z-score
        assert z_scores[-1] == np.max(z_scores)
        assert z_scores[-1] > 3.5  # Should exceed typical threshold

    def test_formula_correctness(self):
        """Verify the formula: Modified Z = 0.6745 * (x - median) / MAD"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        median = np.median(data)  # 5
        mad = calculate_mad(data)  # 2

        z_scores = calculate_modified_zscore(data)

        # Manual calculation for value 9: 0.6745 * (9 - 5) / 2 = 1.349
        expected_z_for_9 = 0.6745 * (9 - median) / mad
        assert np.isclose(z_scores[-1], expected_z_for_9)


class TestDetectOutliersMAD:
    """Tests for detect_outliers_mad function."""

    def test_detects_obvious_outlier(self, skewed_data):
        outlier_mask, z_scores = detect_outliers_mad(skewed_data, threshold=3.5)
        # The value 100 should be flagged as an outlier
        assert outlier_mask[-1] == True
        # Most values should not be outliers
        assert np.sum(outlier_mask) == 1

    def test_no_outliers_in_normal_data(self, sample_data):
        outlier_mask, z_scores = detect_outliers_mad(sample_data, threshold=3.5)
        assert np.sum(outlier_mask) == 0

    def test_threshold_sensitivity(self, skewed_data):
        # Lower threshold should catch more outliers
        mask_strict, _ = detect_outliers_mad(skewed_data, threshold=5.0)
        mask_lenient, _ = detect_outliers_mad(skewed_data, threshold=2.0)

        assert np.sum(mask_lenient) >= np.sum(mask_strict)

    def test_returns_correct_types(self, sample_data):
        outlier_mask, z_scores = detect_outliers_mad(sample_data)
        assert isinstance(outlier_mask, np.ndarray)
        assert isinstance(z_scores, np.ndarray)
        assert outlier_mask.dtype == bool


class TestCalculatePercentiles:
    """Tests for calculate_percentiles function."""

    def test_default_percentiles(self, sample_data):
        percentiles = calculate_percentiles(sample_data)
        assert 'p5' in percentiles
        assert 'p50' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles

    def test_p50_equals_median(self, sample_data):
        percentiles = calculate_percentiles(sample_data)
        assert percentiles['p50'] == calculate_median(sample_data)

    def test_custom_percentiles(self, sample_data):
        percentiles = calculate_percentiles(sample_data, percentiles=[10, 50, 90])
        assert 'p10' in percentiles
        assert 'p50' in percentiles
        assert 'p90' in percentiles
        assert 'p95' not in percentiles

    def test_percentile_ordering(self, sample_data):
        percentiles = calculate_percentiles(sample_data)
        assert percentiles['p5'] <= percentiles['p25']
        assert percentiles['p25'] <= percentiles['p50']
        assert percentiles['p50'] <= percentiles['p75']
        assert percentiles['p75'] <= percentiles['p95']


class TestCalculateIQR:
    """Tests for calculate_iqr function."""

    def test_basic_iqr(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper, iqr = calculate_iqr(data)

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        expected_iqr = q3 - q1

        assert np.isclose(iqr, expected_iqr)
        assert np.isclose(lower, q1 - 1.5 * expected_iqr)
        assert np.isclose(upper, q3 + 1.5 * expected_iqr)

    def test_returns_floats(self, sample_data):
        lower, upper, iqr = calculate_iqr(sample_data)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(iqr, float)


class TestCalculateRobustStatistics:
    """Tests for calculate_robust_statistics function."""

    def test_contains_all_expected_keys(self, sample_data):
        stats = calculate_robust_statistics(sample_data)

        expected_keys = [
            'count', 'median', 'mad', 'mean', 'std',
            'min', 'max', 'iqr', 'iqr_lower_bound', 'iqr_upper_bound',
            'p5', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99',
            'skewness', 'kurtosis'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_count_is_correct(self, sample_data):
        stats = calculate_robust_statistics(sample_data)
        assert stats['count'] == len(sample_data)

    def test_handles_nan_values(self):
        data = np.array([1, 2, np.nan, 4, 5])
        stats = calculate_robust_statistics(data)
        assert stats['count'] == 4  # NaN should be excluded

    def test_skewness_positive_for_right_skewed(self, skewed_data):
        stats = calculate_robust_statistics(skewed_data)
        assert stats['skewness'] > 0


# =============================================================================
# FREQUENCY ANALYZER TESTS
# =============================================================================

class TestFrequencyAnalyzer:
    """Tests for FrequencyAnalyzer class."""

    def test_frequency_table_created(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        freq_table = analyzer.frequency_table

        assert isinstance(freq_table, pd.DataFrame)
        assert 'value' in freq_table.columns
        assert 'count' in freq_table.columns
        assert 'percentage' in freq_table.columns
        assert 'cumulative_percentage' in freq_table.columns

    def test_frequency_counts_correct(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        freq_table = analyzer.frequency_table

        alice_count = freq_table[freq_table['value'] == 'alice']['count'].values[0]
        assert alice_count == 3

    def test_percentages_sum_to_100(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        freq_table = analyzer.frequency_table

        assert np.isclose(freq_table['percentage'].sum(), 100.0)

    def test_cumulative_ends_at_100(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        freq_table = analyzer.frequency_table

        assert np.isclose(freq_table['cumulative_percentage'].iloc[-1], 100.0)

    def test_head_tail_analysis(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        head, tail = analyzer.get_head_tail_analysis(head_pct=50)

        assert len(head) + len(tail) == len(analyzer.frequency_table)

    def test_rare_events(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        rare = analyzer.get_rare_events(threshold=2)

        # charlie has only 1 occurrence, should be rare
        assert 'charlie' in rare['value'].values

    def test_concentration_metrics(self, sample_dataframe):
        analyzer = FrequencyAnalyzer(sample_dataframe, 'user_name')
        metrics = analyzer.get_concentration_metrics()

        assert 'unique_count' in metrics
        assert 'top_1_pct' in metrics
        assert 'gini_coefficient' in metrics
        assert metrics['unique_count'] == 3


# =============================================================================
# THRESHOLD SUGGESTION TESTS
# =============================================================================

class TestSuggestThresholdFromStats:
    """Tests for suggest_threshold_from_stats function."""

    @pytest.fixture
    def sample_stats(self):
        return {
            'median': 50,
            'mad': 10,
            'p90': 80,
            'p95': 90,
            'p99': 100,
            'iqr_upper_bound': 75
        }

    def test_mad_method(self, sample_stats):
        result = suggest_threshold_from_stats(sample_stats, method='mad', sensitivity='medium')

        assert 'threshold' in result
        assert 'formula' in result
        assert result['method'] == 'mad'

        # Verify formula: median + (3.5 * MAD / 0.6745)
        expected = 50 + (3.5 * 10 / 0.6745)
        assert np.isclose(result['threshold'], expected)

    def test_percentile_method(self, sample_stats):
        result = suggest_threshold_from_stats(sample_stats, method='percentile', sensitivity='medium')

        assert result['threshold'] == 90  # P95 for medium sensitivity
        assert 'P95' in result['formula']

    def test_iqr_method(self, sample_stats):
        result = suggest_threshold_from_stats(sample_stats, method='iqr', sensitivity='medium')

        assert result['threshold'] == 75

    def test_sensitivity_affects_threshold(self, sample_stats):
        high = suggest_threshold_from_stats(sample_stats, method='mad', sensitivity='high')
        medium = suggest_threshold_from_stats(sample_stats, method='mad', sensitivity='medium')
        low = suggest_threshold_from_stats(sample_stats, method='mad', sensitivity='low')

        # Higher sensitivity = lower threshold (catches more)
        assert high['threshold'] < medium['threshold'] < low['threshold']


# =============================================================================
# BASELINE DOCUMENTER TESTS
# =============================================================================

class TestBaselineDocumenter:
    """Tests for BaselineDocumenter class."""

    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "baselines"
            documenter = BaselineDocumenter(str(output_dir))
            assert output_dir.exists()

    def test_save_baseline_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            documenter = BaselineDocumenter(tmpdir)
            baseline = DetectionBaseline(
                detection_name="Test Detection",
                detection_id="DET-TEST-001"
            )

            filepath = documenter.save_baseline(baseline, format="json")

            assert Path(filepath).exists()
            assert filepath.endswith(".json")

            with open(filepath) as f:
                saved = json.load(f)
            assert saved['detection_name'] == "Test Detection"

    def test_save_baseline_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            documenter = BaselineDocumenter(tmpdir)
            baseline = DetectionBaseline(
                detection_name="Test Detection",
                detection_id="DET-TEST-001"
            )

            filepath = documenter.save_baseline(baseline, format="markdown")

            assert Path(filepath).exists()
            assert filepath.endswith(".md")

    def test_load_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            documenter = BaselineDocumenter(tmpdir)
            original = DetectionBaseline(
                detection_name="Test Detection",
                detection_id="DET-TEST-001",
                total_records=1000
            )

            filepath = documenter.save_baseline(original, format="json")
            loaded = documenter.load_baseline(filepath)

            assert loaded.detection_name == original.detection_name
            assert loaded.detection_id == original.detection_id
            assert loaded.total_records == original.total_records

    def test_compare_baselines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            documenter = BaselineDocumenter(tmpdir)

            previous = DetectionBaseline(
                detection_name="Test",
                detection_id="DET-001",
                total_records=1000,
                distribution_stats={'median': 50, 'mad': 10}
            )
            current = DetectionBaseline(
                detection_name="Test",
                detection_id="DET-001",
                total_records=1200,
                distribution_stats={'median': 55, 'mad': 12}
            )

            comparison = documenter.compare_baselines(current, previous)

            assert comparison['record_count_change'] == 200
            assert comparison['record_count_change_pct'] == 20.0
            assert 'median' in comparison['stat_changes']


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidateBaselineCoverage:
    """Tests for validate_baseline_coverage function."""

    def test_basic_coverage(self, sample_dataframe):
        result = validate_baseline_coverage(sample_dataframe, 'user_name')

        assert result['total_groups_observed'] == 3
        assert set(result['observed_groups']) == {'alice', 'bob', 'charlie'}

    def test_coverage_with_expected_groups(self, sample_dataframe):
        expected = ['alice', 'bob', 'charlie', 'dave']
        result = validate_baseline_coverage(
            sample_dataframe,
            'user_name',
            expected_groups=expected
        )

        assert 'dave' in result['missing_groups']
        assert result['coverage_pct'] == 75.0  # 3 of 4

    def test_unexpected_groups(self, sample_dataframe):
        expected = ['alice', 'bob']
        result = validate_baseline_coverage(
            sample_dataframe,
            'user_name',
            expected_groups=expected
        )

        assert 'charlie' in result['unexpected_groups']


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestLoadDataFromJson:
    """Tests for load_data_from_json function."""

    def test_load_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {'user': 'alice', 'count': 10},
                {'user': 'bob', 'count': 20}
            ], f)
            f.flush()

            df = load_data_from_json(f.name)

            assert len(df) == 2
            assert 'user' in df.columns
            assert 'count' in df.columns

            Path(f.name).unlink()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_analysis_workflow(self):
        """Test a complete baseline analysis workflow."""
        # Generate synthetic data
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 100)
        outliers = np.array([150, 200])  # Add outliers
        data = np.concatenate([normal_data, outliers])

        # Calculate statistics
        stats = calculate_robust_statistics(data)

        # Detect outliers
        outlier_mask, z_scores = detect_outliers_mad(data, threshold=3.5)

        # Verify outliers detected
        assert np.sum(outlier_mask) >= 2  # At least the injected outliers

        # Verify statistics are reasonable
        assert 40 < stats['median'] < 60  # Should be near 50
        assert stats['skewness'] > 0  # Right-skewed due to outliers

    def test_zscore_threshold_consistency(self):
        """Verify z-score and threshold calculations are consistent."""
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        stats = calculate_robust_statistics(data)

        # Get threshold for z=3.5
        threshold_info = suggest_threshold_from_stats(stats, method='mad', sensitivity='medium')
        threshold = threshold_info['threshold']

        # Calculate z-score for a value at the threshold
        z_at_threshold = 0.6745 * (threshold - stats['median']) / stats['mad']

        # Should be approximately 3.5
        assert np.isclose(z_at_threshold, 3.5, atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
