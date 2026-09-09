"""One-call end-to-end fault detection for testing the
:class:`~energy_fault_detector.fault_detector.FaultDetector` on a single dataset.

The function :func:`~energy_fault_detector.quick_fault_detector` is also exposed as
the ``quick_fault_detector`` command-line entry point.

"""

from .pipeline import quick_fault_detector

__all__ = ["quick_fault_detector"]