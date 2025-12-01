Energy Fault Detector - Autoencoder-based Fault Detection for the Future Energy System
======================================================================================

**Energy Fault Detector** is an open-source Python package designed for the automated detection of anomalies in
operational data from renewable energy systems as well as power grids. It uses autoencoder-based normal behaviour
models to identify irregularities in operational data. In addition to the classic anomaly detection, the package
includes the unique ''ARCANA'' approach for root cause analysis and thus allows interpretable early fault detection.
In addition to the pure ML models, the package also contains a range of preprocessing methods, which are particularly
useful for analyzing systems in the energy sector. A holistic `EnergyFaultDetector` framework is provided for easy use of all
these methods, which can be adapted to the respective use case via a single configuration file.

The software is particularly valuable in the context of the future energy system, optimizing the monitoring and enabling
predictive maintenance of renewable energy assets.

Installation
^^^^^^^^^^^^

To install the `energy-fault-detector` package, run:

.. code-block:: shell

    pip install energy-fault-detector


.. toctree::
    :caption: Contents
    :glob:
    :maxdepth: 2

    usage_examples
    configuration
    logging
    The EnergyFaultDetector package <energy_fault_detector>

Module index
==================

* :ref:`modindex`
