"""Quick energy fault detector CLI tool, to try out the EnergyFaultDetector model on a specific dataset."""

import argparse
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger('energy_fault_detector')
here = Path(__file__).resolve().parent


@dataclass
class Options:
    csv_test_data_path: Optional[str | Path] = None
    train_test_column_name: Optional[str] = None
    train_test_mapping: Optional[dict] = None
    time_column_name: Optional[str] = None
    status_data_column_name: Optional[str] = None
    status_mapping: Optional[dict] = None
    status_label_confidence_percentage: float = 0.95
    min_anomaly_length: int = 18
    features_to_exclude: List[str] = field(default_factory=list)
    angle_features: List[str] = field(default_factory=list)
    automatic_optimization: bool = True
    enable_debug_plots: bool = False


def load_options_from_yaml(file_path: str | Path) -> Options:
    """Load options from a YAML file and return an Options dataclass."""
    with open(file_path, 'r') as file:
        options_dict = yaml.safe_load(file)
        return Options(**options_dict)


def main():
    parser = argparse.ArgumentParser(
        description='''
        Quick Fault Detection Tool for Energy Systems. This tool analyzes provided data using an
        autoencoder-based approach to identify anomalies based on learned normal behavior.
        Anomalies are then aggregated into events for further analysis.

        Required Arguments:
        - csv_data_path: Path to a CSV file containing training data.

        Optional Arguments (via YAML file):
        - options: Path to a YAML file containing additional options.

        Example YAML file structure:
            csv_test_data_path: "path/to/test_data.csv"
            train_test_column_name: "train_test"      # true = training data
            train_test_mapping:
                train: true
                test: false
            time_column_name: "timestamp"
            status_data_column_name: "status"         # true = normal behaviour
            status_mapping: 
                production: true
                service: false
                error: false
            status_label_confidence_percentage: 0.95  # contamination level
            min_anomaly_length: 18
            features_to_exclude: 
              - do_not_use_this_feature_1
              - do_not_use_this_feature_2
            angle_features:
              - angle1
              - angle2
            automatic_optimization: true
            enable_debug_plots: false
        ''',

        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'csv_data_path',
        type=Path,
        help='Path to a CSV file containing training data.'
    )
    parser.add_argument(
        '--options',
        type=Path,
        help='Path to a YAML file containing additional options.',
        default=None,
        required=False,
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        help='Path to a directory where results will be saved.',
        default='results'
    )
    parser.add_argument(
        '--c2c_example',
        action='store_true',
        help='Whether to use default settings for a CARE2Compare dataset.',
    )

    args = parser.parse_args()
    logger.info(f"CSV Data Path: {args.csv_data_path}")

    if args.c2c_example:
        logger.info("Using default settings for CARE2Compare dataset.")
    else:
        logger.info(f"Options YAML: {args.options}")

    logger.info(f"Results Directory: {args.results_dir}")
    args.results_dir.mkdir(exist_ok=True)

    options = Options()  # Initialize with default values
    if args.options:
        options = load_options_from_yaml(args.options)
    elif args.c2c_example:
        options = load_options_from_yaml(here / 'c2c_options.yaml')

    print(options)

    # Call the quick_fault_detector function with parsed arguments
    try:
        from .quick_fault_detection import quick_fault_detector
        prediction_results, event_meta_data = quick_fault_detector(
            csv_data_path=args.csv_data_path,
            csv_test_data_path=options.csv_test_data_path,
            train_test_column_name=options.train_test_column_name,
            train_test_mapping=options.train_test_mapping,
            time_column_name=options.time_column_name,
            status_data_column_name=options.status_data_column_name,
            status_mapping=options.status_mapping,
            status_label_confidence_percentage=options.status_label_confidence_percentage,
            features_to_exclude=options.features_to_exclude,
            angle_features=options.angle_features,
            automatic_optimization=options.automatic_optimization,
            enable_debug_plots=options.enable_debug_plots,
            min_anomaly_length=options.min_anomaly_length,
            save_dir=args.results_dir,
        )
        logger.info(f'Fault detection completed. Results are saved in {args.results_dir}.')
        prediction_results.save(args.results_dir)
        event_meta_data.to_csv(args.results_dir / 'events.csv', index=False)

    except Exception as e:
        logger.error(f'An error occurred: {e}')
        raise


if __name__ == '__main__':
    main()
