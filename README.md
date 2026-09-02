
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AEFDI/EnergyFaultDetector/main/img/2025_Logo_Energy-Fault-Detector_white-green.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AEFDI/EnergyFaultDetector/main/img/Logo_Energy-Fault-Detector.png">
  <img alt="EnergyFaultDetector Logo" src="https://raw.githubusercontent.com/AEFDI/EnergyFaultDetector/main/img/Logo_Energy-Fault-Detector.png" height="100">
</picture>


# Energy Fault Detector - Autoencoder-based Fault Detection for the Future Energy System

[![Python](https://img.shields.io/pypi/pyversions/energy-fault-detector)](https://pypi.org/project/energy-fault-detector/)
[![PyPI version](https://img.shields.io/pypi/v/energy-fault-detector)](https://pypi.org/project/energy-fault-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AEFDI/EnergyFaultDetector/actions/workflows/run-tests.yml/badge.svg)](https://github.com/AEFDI/EnergyFaultDetector/actions/workflows/run-tests.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://aefdi.github.io/EnergyFaultDetector/)

**Energy Fault Detector** is an open-source Python package for automated anomaly detection in operational data from
renewable energy systems and power grids. It uses autoencoder-based normal-behaviour models to identify 
irregularities and includes the [ARCANA](https://doi.org/10.1016/j.egyai.2021.100065) method for interpretable 
root cause analysis.

<img src="https://raw.githubusercontent.com/AEFDI/EnergyFaultDetector/main/img/OSS-Grafical_abstract2.png" alt="drawing" width="600" style="display: block; margin: 0 auto" />

## Features (at a glance)
- **User-friendly interface**: One-command CLI (`quick_fault_detector`) demo and a simple Python API.
- Built-in **data preprocessing** (clipping, imputation, counter→rate, angle transforms, etc.)
- **Fault Detection**: Autoencoder-based **normal behaviour modelling** for time series and tabular data.
- **Root Cause Analysis**: Pinpoints the specific sensor values responsible for detected anomalies using [ARCANA](https://doi.org/10.1016/j.egyai.2021.100065).
- **Scalability**: Algorithms can easily be adapted to various datasets and trained models can be transferred to and
   fine-tuned on similar datasets. Quickly evaluate many different model configurations
- Support for **benchmark datasets** (CARE2Compare, PreDist) and CARE-Score evaluation

See the [online documentation](https://aefdi.github.io/EnergyFaultDetector/) for concepts, model types, and
configuration details.

## Installation
```bash
pip install energy-fault-detector
```
Requirements: Python 3.10–3.12, TensorFlow ≥ 2.15

For development (tests, linting):
```bash
pip install energy-fault-detector[dev]
```

## Quickstart

```python
import pandas as pd
from energy_fault_detector import FaultDetector, Config
from energy_fault_detector.config import generate_quickstart_config

### 1. Load your data
df = pd.read_csv("my_data.csv", parse_dates=["timestamp"], index_col="timestamp")
sensor_data = df[["power", "wind_speed", "pitch"]]
normal_index = df["status"] == "normal"  # optional boolean mask

### 2. Generate a default configuration
config = generate_quickstart_config()

### 3. Train a normal-behaviour model
fault_detector = FaultDetector(config=config, model_directory="my_model")
fault_detector.fit(sensor_data=sensor_data, normal_index=normal_index)

### 4. Predict anomalies
results = fault_detector.predict(sensor_data=sensor_data)

print(results.predicted_anomalies.sum(), "anomalies detected")
```

More examples: [Usage examples](https://aefdi.github.io/EnergyFaultDetector/usage_examples.html).

## Quick Fault Detection (CLI)
Run the full pipeline (train → predict → events → ARCANA) in a single command:
```bash
quick_fault_detector path/to/data.csv
```
For [CARE2Compare](https://doi.org/10.5281/zenodo.10958774) data:
```bash
quick_fault_detector path/to/c2c_dataset.csv --c2c_example
```
The CLI saves plots and CSV results to a results directory.
See the [CLI documentation](https://aefdi.github.io/EnergyFaultDetector/quick_fault_detection.html) for details.

## Documentation
Full docs (concepts, available models, configuration reference, evaluation, and examples): [https://aefdi.github.io/EnergyFaultDetector/](https://aefdi.github.io/EnergyFaultDetector/)

## Examples and notebooks

The repository contains Jupyter notebooks with end-to-end examples and evaluation workflows in the
[`notebooks/`](./notebooks) folder, for example:

- Quick fault detection on a CSV file
- Standard `FaultDetector` training and prediction
- Sequence models (LSTM/CNN) on time-series data
- CARE2Compare and PreDist benchmark evaluations

These notebooks complement the documentation and are a good starting point for interactive exploration.

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

### Planned updates and features

- Demo: Easy demo website
- Extending the core models:
   - Variational autoencoders
   - Model ensembles
- Unification, standardisation and generic improvements
   - Data preparation (e.g. extend imputation strategies).
   - No or low configuration need (e.g. use defaults where possible).
   - Upgrade to Keras 3.0
- Root cause analysis expansion
   - integrate SHAP and possibly other xAI-methods.
- Integrations
   - logging/tracking to MLFlow for hyperparameter tuning and easy model deployment.
   - Edge deployment

## License
This project is licensed under the [MIT License](./LICENSE).

## Background
This project was initially developed by the research team AEFDI at the Fraunhofer IEE in the research project ADWENTURE (funded by the German Federal Ministry for Economic Affairs and Climate Action (BMWK)), 
to create a software for early fault detection in wind turbines. The software was developed in such a way that the algorithms do not depend on a specific data source
and can be applied to other use cases as well.

## References
If you use this work, please cite us:

**Fault detection in district heating substations**:
- Enabling Predictive Maintenance in District Heating Substations: A Labelled Dataset and Fault Detection Evaluation Framework based on Service Data. Energy. 2026;355:141178. https://doi.org/10.1016/j.energy.2026.141178
- Dataset: PreDist Dataset - Operational data of district heating substations labelled with faults and maintenance information. Zenodo, Nov 2025, https://doi.org/10.5281/zenodo.17522254.

**ARCANA Algorithm**:
Autoencoder-based anomaly root cause analysis for wind turbines. Energy and AI. 2021;4:100065. https://doi.org/10.1016/j.egyai.2021.100065

**CARE to Compare dataset and CARE-Score**:
- Paper: CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data. Data. 2024; 9(12):138. https://doi.org/10.3390/data9120138 
- Dataset: Wind Turbine SCADA Data For Early Fault Detection. Zenodo, Oct. 2024, https://doi.org/10.5281/ZENODO.14958989.

**Transfer learning methods**:
Transfer learning applications for autoencoder-based anomaly detection in wind turbines. Energy and AI. 2024;17:100373. https://doi.org/10.1016/j.egyai.2024.100373

**Autoencoder-based anomaly detection**:
Evaluation of Anomaly Detection of an Autoencoder Based on Maintenance Information and Scada-Data. Energies. 2020; 13(5):1063., https://doi.org/10.3390/en13051063.

## Contact
For questions, feedback, or support integrating the EnergyFaultDetector into your operations, please contact [aefdi@iee.fraunhofer.de](mailto:aefdi@iee.fraunhofer.de).
