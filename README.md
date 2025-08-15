
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/2025_Logo_Energy-Fault-Detector_white-green.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/Logo_Energy-Fault-Detector.png">
  <img alt="EnergyFaultDetector Logo" src="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/Logo_Energy-Fault-Detector.png" height="100">
</picture>


# Energy Fault Detector - Autoencoder-based Fault Detection for the Future Energy System

**Energy Fault Detector** is an open-source Python package designed for the automated detection of anomalies in
operational data from renewable energy systems as well as power grids. It uses autoencoder-based normal behaviour
models to identify irregularities in operational data. In addition to the classic anomaly detection, the package 
includes the unique “ARCANA” approach for root cause analysis and thus allows interpretable early fault detection. 
In addition to the pure ML models, the package also contains a range of preprocessing methods, which are particularly 
useful for analyzing systems in the energy sector. A holistic `EnergyFaultDetector` framework is provided for easy use of all 
these methods, which can be adapted to the respective use case via a single configuration file.

The software is particularly valuable in the context of the future energy system, optimizing the monitoring and enabling
predictive maintenance of renewable energy assets.

<img src="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/OSS-Grafical_abstract2.png" alt="drawing" width="600" style="display: block; margin: 0 auto" />

## Main Features
- **User-friendly interface**: Easy to use and quick to demo using the [command line interface](#Quick-fault-detection).
- **Data Preprocessing Module**: Prepares numerical operational data for analysis with the `EnergyFaultDetector`, 
  with many options such as data clipping, imputation, signal hangers and column selection based on variance and
  missing values. 
- **Fault Detection**: Uses autoencoder architectures to model normal operational behavior and identify deviations.
- **Root Cause Analysis**: Pinpoints the specific sensor values responsible for detected anomalies using [ARCANA](https://doi.org/10.1016/j.egyai.2021.100065).
- **Scalability**: Algorithms can easily be adapted to various datasets and trained models can be transferred to and
   fine-tuned on similar datasets. Quickly evaluate many different model configurations

## Installation
To install the `energy-fault-detector` package, run: `pip install energy-fault-detector`


## Quick fault detection
For a quick demo on a specific dataset, run:

```quick_fault_detector <path_to_your_dataset.csv>```

For more options, run ```quick_fault_detector -h```.

For an example using one of the CARE2Compare datasets, run:
```quick_fault_detector <path_to_c2c_dataset.csv> --c2c_example```

For more information, have a look at the notebook [Quick Failure Detection](./notebooks/Example%20-%20Quick%20Failure%20Detection.ipynb)


## Fault detection in 5 lines of code

```python
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.config import Config

fault_detector = FaultDetector(config=Config('base_config.yaml'))
model_data = fault_detector.train(sensor_data=sensor_data, normal_index=normal_index)
results = fault_detector.predict(sensor_data=test_sensor_data)
```

The pandas `DataFrame` `sensor_data` contains the operational data in wide format with the timestamp as index, the
pandas `Series` `normal_index` indicates which timestamps are considered 'normal' operation and can be used to create
a normal behaviour model. The [`base_config.yaml`](energy_fault_detector/base_config.yaml) file contains all model 
settings, an example is found [here](energy_fault_detector/base_config.yaml).


## Background
This project was initially developed in the research project ADWENTURE, to create a software for early fault detection
in wind turbines. The software was developed in such a way that the algorithms do not depend on a specific data source
and can be applied to other use cases as well.

## Documentation
Comprehensive documentation is available [here](https://aefdi.github.io/EnergyFaultDetector/).

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

### Planned updates and features
1. More autoencoder types:
   1. Variational autoencoders
   2. CNN- and LSTM-based autoencoders with time-series support.

2. Unification, standardisation and generic improvements
   1. Additional options for all autoencoders (e.g. drop out, regularization)
   2. Data preparation (e.g. extend imputation strategies).
   3. Download method for the Care2Compare class.
   3. Unify default value settings. 
   4. No or low configuration

3. Conditions and dependency updates
   1. Conditional autoencoder 
   2. Upgrade to Keras 3.0

4. Root cause analysis expansion: integrate SHAP and possibly other XAI-methods.

## License
This project is licensed under the [MIT License](./LICENSE).

## References
If you use this work, please cite us:

**ARCANA Algorithm**:
Autoencoder-based anomaly root cause analysis for wind turbines. Energy and AI. 2021;4:100065. https://doi.org/10.1016/j.egyai.2021.100065

**CARE to Compare dataset and CARE-Score**:
- Paper: CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data. Data. 2024; 9(12):138. https://doi.org/10.3390/data9120138 
- Dataset: Wind Turbine SCADA Data For Early Fault Detection. Zenodo, Mar. 2025, https://doi.org/10.5281/ZENODO.14958989.

**Transfer learning methods**:
Transfer learning applications for autoencoder-based anomaly detection in wind turbines. Energy and AI. 2024;17:100373. https://doi.org/10.1016/j.egyai.2024.100373

**Autoencoder-based anomaly detection**:
Evaluation of Anomaly Detection of an Autoencoder Based on Maintenance Information and Scada-Data. Energies. 2020; 13(5):1063., https://doi.org/10.3390/en13051063.
