import unittest
import numpy as np

from energy_fault_detector.data_splitting.data_gap_handler import shift_array, DataGapHandler


class TestShiftArray(unittest.TestCase):
    """Unit tests for the shift_array utility."""

    def test_shift_array_zero(self) -> None:
        arr = np.array([1, 2, 3, 4])
        result = shift_array(arr, num=0, fill_value=-1)
        np.testing.assert_array_equal(result, arr)

    def test_shift_array_positive(self) -> None:
        arr = np.array([1, 2, 3, 4])
        result = shift_array(arr, num=2, fill_value=0)
        expected = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_shift_array_negative(self) -> None:
        arr = np.array([1, 2, 3, 4])
        result = shift_array(arr, num=-1, fill_value=9)
        expected = np.array([2, 3, 4, 9])
        np.testing.assert_array_equal(result, expected)


class TestDataGapHandler(unittest.TestCase):
    """Unit tests for DataGapHandler."""

    def setUp(self) -> None:
        # Base regular timestamps every 10 minutes
        self.freq = np.timedelta64(10, "m")
        self.base_times = np.array(
            [np.datetime64("2025-01-01T00:00") + i * self.freq for i in range(10)]
        )

    def test_no_gaps(self) -> None:
        handler = DataGapHandler(self.base_times, freq=self.freq)

        self.assertIsNone(handler.data_gaps)
        # No timestamp is in a gap
        for ts in self.base_times:
            self.assertFalse(handler.is_in_gap(ts))

        # No interval has gaps
        self.assertFalse(
            handler.has_data_gaps(self.base_times[0], self.base_times[-1])
        )
        # No next gap
        self.assertIsNone(handler.get_next_gap_after(self.base_times[0]))

    def test_single_gap_detection(self) -> None:
        # Introduce a single gap: between index 4 and 5, jump by 1 hour
        times_with_gap = self.base_times.copy()
        times_with_gap[5:] = (
            times_with_gap[4] + np.timedelta64(60, "m")
            + (np.arange(len(times_with_gap) - 5) * self.freq)

        )

        handler = DataGapHandler(times_with_gap, freq=self.freq)
        self.assertIsNotNone(handler.data_gaps)
        self.assertEqual(handler.data_gaps.shape[0], 1)

        gap_start, gap_end = handler.data_gaps[0]
        # Gap should be between the timestamps around index 4 and 5
        self.assertEqual(gap_start, times_with_gap[4])
        self.assertEqual(gap_end, times_with_gap[5])

        # Timestamps between gap_start and gap_end are in gap
        self.assertTrue(handler.is_in_gap(gap_start))
        self.assertTrue(handler.is_in_gap(gap_end))
        # A timestamp before the gap is not in gap
        self.assertFalse(handler.is_in_gap(times_with_gap[3]))
        # A timestamp after the gap is not in gap
        self.assertFalse(handler.is_in_gap(times_with_gap[-1]))

        # Interval overlapping the gap should report gaps
        self.assertTrue(handler.has_data_gaps(times_with_gap[0], times_with_gap[-1]))
        # Interval completely before the gap
        self.assertFalse(handler.has_data_gaps(times_with_gap[0], times_with_gap[3]))
        # Interval completely after the gap
        self.assertFalse(handler.has_data_gaps(times_with_gap[6], times_with_gap[-1]))

        # Next gap after a timestamp before the gap
        next_gap = handler.get_next_gap_after(times_with_gap[0])
        self.assertEqual(next_gap, (gap_start, gap_end))

        # No next gap after the gap end
        self.assertIsNone(handler.get_next_gap_after(gap_end))

    def test_multiple_gaps_sorted_and_queries(self) -> None:
        # Create two gaps:
        # 1) between indices 2 and 3
        # 2) between indices 6 and 7
        times = self.base_times.copy()
        # First gap: add 1 hour at index 3 onward
        times[3:] = (
            times[2] + np.timedelta64(60, "m")
            + (np.arange(len(times) - 3) * self.freq)

        )
        # Second gap: add another 2 hours at index 7 onward
        times[7:] = (
            times[6] + np.timedelta64(120, "m")
            + (np.arange(len(times) - 7) * self.freq)

        )

        handler = DataGapHandler(times, freq=self.freq)
        self.assertIsNotNone(handler.data_gaps)
        self.assertEqual(handler.data_gaps.shape[0], 2)

        # Ensure gaps are sorted by start time
        gap_starts = handler.data_gaps[:, 0]
        self.assertTrue(np.all(gap_starts[:-1] <= gap_starts[1:]))

        # Check is_in_gap across both gaps
        for gap_start, gap_end in handler.data_gaps:
            self.assertTrue(handler.is_in_gap(gap_start))
            self.assertTrue(handler.is_in_gap(gap_end))

        # has_data_gaps across whole range is True
        self.assertTrue(handler.has_data_gaps(times[0], times[-1]))

        # get_next_gap_after returns the first then second gap
        first_gap = handler.data_gaps[0]
        second_gap = handler.data_gaps[1]
        self.assertEqual(handler.get_next_gap_after(times[0]), (first_gap[0], first_gap[1]))
        self.assertEqual(handler.get_next_gap_after(first_gap[1]), (second_gap[0], second_gap[1]))
        self.assertIsNone(handler.get_next_gap_after(second_gap[1]))
