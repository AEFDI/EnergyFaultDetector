Logging Configuration
=====================

The framework uses Python's built-in logging module for logging.
You can customize this configuration to suit your needs.

Default Configuration
---------------------

The framework uses a default logging configuration file ``energy_fault_detector/logging.yaml``.
The logger used throughout the code is called ``energy_fault_detector``.

You can silence the logger as follows:

.. code-block:: python
    import logging
    from energy_fault_detector.fault_detector import FaultDetector, Config

    logger = logging.getLogger('energy_fault_detector')
    logger.setLevel(logging.CRITICAL)


Customizing Logging
-------------------

You can specify your own logging configuration file by setting an environment variable before running the application.

To specify a custom logging configuration, set the ``LOG_CFG`` environment variable to the path of your configuration file:

.. code-block:: bash

    export LOG_CFG="/path/to/your_logging.yaml"

or
.. code-block:: python

    import os

    os.environ['LOG_CFG'] = '/path/to/your_logging.yaml'
