import numpy as np
import pandas as pd
import pytest

from energy_fault_detector.utils.analysis import calculate_criticality, create_events


# ──────────────────────────────────────────────────────────────────────────────
# calculate_criticality
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateCriticalitySingleIndex:
    """Existing behavior with a simple DatetimeIndex."""

    def test_all_normal_no_anomalies(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="10min")
        anomalies = pd.Series([False] * 5, index=idx)
        crit = calculate_criticality(anomalies)
        assert (crit == 0).all()

    def test_all_anomalies(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="10min")
        anomalies = pd.Series([True] * 5, index=idx)
        crit = calculate_criticality(anomalies)
        assert list(crit) == [1, 2, 3, 4, 5]

    def test_criticality_decreases(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="10min")
        anomalies = pd.Series([True, True, True, False, False, False], index=idx)
        crit = calculate_criticality(anomalies)
        assert list(crit) == [1, 2, 3, 2, 1, 0]

    def test_max_criticality_cap(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="10min")
        anomalies = pd.Series([True] * 5, index=idx)
        crit = calculate_criticality(anomalies, max_criticality=3)
        assert list(crit) == [1, 2, 3, 3, 3]

    def test_non_normal_periods_ignored(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="10min")
        anomalies = pd.Series([True, True, True, True, True], index=idx)
        normal_idx = pd.Series([True, True, False, False, True], index=idx)
        crit = calculate_criticality(anomalies, normal_idx=normal_idx)
        # +1, +1, 0 (not normal), 0 (not normal), +1
        assert list(crit) == [1, 2, 2, 2, 3]


class TestCalculateCriticalityMultiIndex:
    """MultiIndex (asset_id, timestamp) behavior."""

    @pytest.fixture
    def multi_index_data(self):
        """Two assets, each with 5 timestamps."""
        times = pd.date_range("2024-01-01", periods=5, freq="10min")
        idx = pd.MultiIndex.from_product(
            [["asset_A", "asset_B"], times], names=["asset_id", "timestamp"]
        )
        return idx, times

    def test_groups_are_independent(self, multi_index_data):
        """Criticality in asset_A does not leak into asset_B."""
        idx, times = multi_index_data
        # asset_A: all anomalies → criticality rises
        # asset_B: no anomalies → criticality stays 0
        anomalies = pd.Series(
            [True] * 5 + [False] * 5, index=idx
        )
        crit = calculate_criticality(anomalies)

        assert list(crit.loc["asset_A"]) == [1, 2, 3, 4, 5]
        assert list(crit.loc["asset_B"]) == [0, 0, 0, 0, 0]

    def test_multiindex_with_normal_idx(self, multi_index_data):
        """Normal index is respected per group."""
        idx, times = multi_index_data
        anomalies = pd.Series([True] * 10, index=idx)
        # asset_A: normal everywhere, asset_B: not normal at indices 0,1
        normal_idx = pd.Series(
            [True] * 5 + [False, False, True, True, True], index=idx
        )
        crit = calculate_criticality(anomalies, normal_idx=normal_idx)

        assert list(crit.loc["asset_A"]) == [1, 2, 3, 4, 5]
        # asset_B: 0, 0 (not normal), then +1, +1, +1
        assert list(crit.loc["asset_B"]) == [0, 0, 1, 2, 3]

    def test_multiindex_init_criticality_per_group(self, multi_index_data):
        """init_criticality applies to each group independently."""
        idx, _ = multi_index_data
        anomalies = pd.Series([False] * 10, index=idx)
        crit = calculate_criticality(anomalies, init_criticality=3)
        # Each group starts at 3 and decreases
        assert list(crit.loc["asset_A"]) == [2, 1, 0, 0, 0]
        assert list(crit.loc["asset_B"]) == [2, 1, 0, 0, 0]

    def test_preserves_multiindex(self, multi_index_data):
        """Output has the same MultiIndex as input."""
        idx, _ = multi_index_data
        anomalies = pd.Series([True] * 10, index=idx)
        crit = calculate_criticality(anomalies)
        assert isinstance(crit.index, pd.MultiIndex)
        assert crit.index.equals(idx)


# ──────────────────────────────────────────────────────────────────────────────
# create_events
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateEventsSingleIndex:
    """Existing behavior with a simple DatetimeIndex."""

    def test_single_event(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="10min")
        df = pd.DataFrame({"power": np.random.randn(20)}, index=idx)
        bools = pd.Series([False] * 5 + [True] * 12 + [False] * 3, index=idx)
        meta, events = create_events(df, bools, min_event_length=10)
        assert len(meta) == 1
        assert len(events) == 1
        assert len(events[0]) == 12

    def test_no_events_below_min_length(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="10min")
        df = pd.DataFrame({"power": np.random.randn(10)}, index=idx)
        bools = pd.Series([True] * 3 + [False] * 7, index=idx)
        meta, events = create_events(df, bools, min_event_length=5)
        assert len(meta) == 0
        assert len(events) == 0

    def test_multiple_events(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="10min")
        df = pd.DataFrame({"power": np.random.randn(30)}, index=idx)
        bools = pd.Series(
            [True] * 10 + [False] * 5 + [True] * 10 + [False] * 5, index=idx
        )
        meta, events = create_events(df, bools, min_event_length=5)
        assert len(meta) == 2
        assert len(events) == 2


class TestCreateEventsMultiIndex:
    """MultiIndex (asset_id, timestamp) behavior."""

    @pytest.fixture
    def multi_index_sensor_data(self):
        times = pd.date_range("2024-01-01", periods=20, freq="10min")
        idx = pd.MultiIndex.from_product(
            [["asset_A", "asset_B"], times], names=["asset_id", "timestamp"]
        )
        df = pd.DataFrame({"power": np.random.randn(40)}, index=idx)
        return df, idx, times

    def test_events_per_group(self, multi_index_sensor_data):
        """Events are detected independently per asset."""
        df, idx, times = multi_index_sensor_data
        # asset_A: event from index 5-16 (12 timestamps)
        # asset_B: no event (only 3 consecutive True)
        bools_a = [False] * 5 + [True] * 12 + [False] * 3
        bools_b = [False] * 8 + [True] * 3 + [False] * 9
        bools = pd.Series(bools_a + bools_b, index=idx)

        meta, events = create_events(df, bools, min_event_length=10)

        assert len(meta) == 1
        assert meta.iloc[0]["group"] == "asset_A"
        assert len(events) == 1

    def test_events_in_multiple_groups(self, multi_index_sensor_data):
        """Events detected in both assets."""
        df, idx, times = multi_index_sensor_data
        # Both assets have a 12-timestamp event
        bools_a = [False] * 5 + [True] * 12 + [False] * 3
        bools_b = [False] * 2 + [True] * 12 + [False] * 6
        bools = pd.Series(bools_a + bools_b, index=idx)

        meta, events = create_events(df, bools, min_event_length=10)

        assert len(meta) == 2
        assert set(meta["group"]) == {"asset_A", "asset_B"}
        assert len(events) == 2

    def test_no_events_returns_empty(self, multi_index_sensor_data):
        """No events in any group."""
        df, idx, _ = multi_index_sensor_data
        bools = pd.Series([False] * 40, index=idx)
        meta, events = create_events(df, bools, min_event_length=10)
        assert meta.empty
        assert len(events) == 0

    def test_meta_has_group_column(self, multi_index_sensor_data):
        """MultiIndex results include a 'group' column."""
        df, idx, _ = multi_index_sensor_data
        bools = pd.Series([True] * 20 + [False] * 20, index=idx)
        meta, events = create_events(df, bools, min_event_length=10)
        assert "group" in meta.columns
