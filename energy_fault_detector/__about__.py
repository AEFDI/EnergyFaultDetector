from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "energy-fault-detector"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:
    # Fallback for running from a git checkout without installation
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "0.0.0"

__copyright__ = "2025, Fraunhofer IEE"
__title__ = "EnergyFaultDetector"
__description__ = ("Energy Fault Detector is an open-source Python package designed for the automated detection of"
                   " anomalies in operational data from renewable energy systems as well as power grids.")
__author__ = "AEFDI"
