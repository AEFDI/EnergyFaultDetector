
from typing import Optional, Dict, Tuple, List, Any, Union
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix

from energy_fault_detector.utils._deprecations import deprecate_kwargs  # pylint: disable=protected-access
from energy_fault_detector.utils.analysis import calculate_criticality

logger = logging.getLogger('energy_fault_detector')

# deprecations - map old arg names to new arg names
mapping = {
    "min_fraction_anomalous_timestamps": "min_fraction_anomalous",
    "eventwise_f_score_beta": "reliability_beta",
    "weighted_score_w": "earliness_w",
    "eventwise_f_score_w": "reliability_w"
}


class CAREScore:
    """Calculate the CARE score for early fault-detection algorithms.

    The CARE score combines Coverage, Accuracy, Reliability and Earliness to evaluate early fault-detection performance
    (see CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data,
     https://doi.org/10.3390/data9120138). The goal of the CARE-Score is to evaluate the ability of a given model to
    separate `normal behavior` from `actionable anomalies` (see glossary for definitions), that lead to a fault or
    indicate a fault.

    Usage: 
        For each event in your dataset call `evaluate_event`. Afterward, call `get_final_score` to calculate the
        CARE-score of your model.

    Requirements: 
        For calculating the CARE-Score presence of at least one evaluated `anomaly-event` and at least one
        evaluated `normal event` (see glossary below) is required.

    Glossary:
        - normal behavior: Data points representing expected system behavior.
        - actionable anomaly: Unexpected system behavior where maintenance could prevent a fault.
        - non-actionable anomaly: Neither normal behavior nor actionable anomaly (e.g. turbine shut down due to
          maintenance actions).
        - anomaly event: A sequence labeled with actionable anomalies within the prediction window.
        - normal event: A sequence containing normal behavior within the prediction window.
        - normal_index: Boolean mask separating normal behavior (True) from non-actionable anomaly (False).
        - Coverage: Pointwise F-score on anomaly-events.
        - Accuracy: Accuracy on normal-events.
        - Reliability: Eventwise F-score across evaluated events.
        - Earliness: Weighted score measuring how early an anomaly is detected.
        - criticality: Counting measure applied to the prediction of anomaly detection models.
          Used for criticality-based detection. For a detailed explanation see Algorithm 1 in the paper
          (https://doi.org/10.3390/data9120138).
          Choose a criticality threshold based on application needs: e.g. how many anomalous points need to be observed
          in advance to be actionable? After how many anomalous data points is an anomaly significant?
    """

    @deprecate_kwargs(mapping, prefer="old")
    def __init__(self, coverage_beta: float = 0.5, reliability_beta: float = 0.5, coverage_w: float = 1.,
                 accuracy_w: float = 2., earliness_w: float = 1., reliability_w: float = 1.,
                 criticality_threshold: int = 72, min_fraction_anomalous: float = 0.1,
                 ws_start_of_descend: Tuple[int, int] = (1, 4), anomaly_detection_method: str = 'criticality',
                 *,  # deprecated (keyword-only) names:
                 min_fraction_anomalous_timestamps: float = None, eventwise_f_score_beta: float = None,
                 weighted_score_w: float = None, eventwise_f_score_w: float = None):
        """Initialize CAREScore.

        Args:
            coverage_beta (float): Beta parameter for Coverage (pointwise F-score). Default: 0.5.
            reliability_beta (float): Beta parameter for Reliability (event-wise F-score). Default: 0.5.
            coverage_w (float): Weight for Coverage (point-wise F-Score) in the final CARE-score. Default: 1.0.
            accuracy_w (float): Weight for Accuracy in the final CARE-score. Default: 2.0.
            reliability_w (float): Weight for Reliability (eventwise F-score) in the final CARE-score. Default: 1.0.
            earliness_w (float): Weight for Earliness (weighted score) in the final CARE-score. Default: 1.0.
            anomaly_detection_method (str): Method used to calculate anomaly detection score. Either 'criticality' or
                'fraction'. Default: 'criticality'.
            criticality_threshold (int): Threshold for criticality-based detection. Default: 72.
                If the criticality exceeds this threshold, the event will be detected as an anomaly.
            min_fraction_anomalous (float): Threshold for fraction-based detection. Default: 0.1.
                If the fraction of event data points exceeds this threshold, the event will be detected as an anomaly.
            ws_start_of_descend (Tuple[int, int]): Fraction (numerator, denominator) where weights start to decay for
                the Earliness-Score (weighted score). Default is (1, 4).

        Raises:
            ValueError: If anomaly_detection_method is not 'criticality' or 'fraction'.
        """

        if anomaly_detection_method not in ['criticality', 'fraction']:
            raise ValueError("Anomaly detection method must be either 'criticality' or 'fraction'")

        self.coverage_beta = coverage_beta
        self.reliability_beta = reliability_beta

        self.coverage_w = coverage_w
        self.accuracy_w = accuracy_w
        self.earliness_w = earliness_w
        self.reliability_w = reliability_w

        self.min_fraction_anomalous = min_fraction_anomalous
        self.criticality_threshold = criticality_threshold
        self.anomaly_detection_method = anomaly_detection_method

        self.ws_start_of_descend = ws_start_of_descend

        self._evaluated_events: List[Dict[str, Any]] = []

    @property
    def evaluated_events(self) -> pd.DataFrame:
        """Return a DataFrame with evaluated events.

        The DataFrame is built from the internal evaluated-events list. If the DataFrame is non-empty,
        an additional column 'anomaly_detected' is computed:

        - If anomaly_detection_method == 'criticality': anomaly_detected = max_criticality >= criticality_threshold
        - Else (fraction-based): anomaly_detected = (tp + fp) / (tp + fp + fn + tn) >= min_fraction_anomalous

        Returns:
            pd.DataFrame: DataFrame containing evaluated event records. Expected columns include:
                - event_id (int)
                - event_label (str)
                - weighted_score (float)
                - max_criticality (float)
                - tp, fp, tn, fn (ints)
                - f_beta_score, accuracy (floats)
                - anomaly_detected (bool) — added as described above.
        """
        df = pd.DataFrame(self._evaluated_events)
        if not df.empty:
            if self.anomaly_detection_method == 'criticality':
                df['anomaly_detected'] = df['max_criticality'] >= self.criticality_threshold
            else:
                df['anomaly_detected'] = (
                    # predicted positive / total event length
                    (df['tp'] + df['fp']) / (df['tp'] + df['fp'] + df['fn'] + df['tn'])
                ) >= self.min_fraction_anomalous
        return df

    def evaluate_event(self, event_start: Union[int, pd.Timestamp], event_end: Union[int, pd.Timestamp],
                       event_label: str, predicted_anomalies: pd.Series, normal_index: Optional[pd.Series] = None,
                       evaluate_until_event_end: Union[str, bool] = False, event_id: Optional[int] = None,
                       ignore_normal_index: bool = False) -> Dict[str, Union[float, int]]:
        """Evaluate the prediction of a fault detection model for a single event.

        If a normal_index is provided, metrics are computed only for timestamps where normal_index is True
        (unless ignore_normal_index is True).

        The argument `evaluate_until_event_end` determines which part of the provided data is used for
        evaluation. It might be useful to set this to True or `anomaly_only` if you expect normal behaviour may change
        after a fault.

        Args:
            event_start (int, pd.Timestamp): Start index/timestamp of the event.
            event_end (int, pd.Timestamp): End index/timestamp of the event.
            event_label (str): True label of the event. This can be either 'anomaly' or 'normal'.
            predicted_anomalies (pd.Series): Boolean pandas series, indicating whether an anomaly was detected.
                Index must match the data type of `event_start` and `event_end`.
            normal_index (pd.Series, optional): Boolean mask marking normal operation (True) vs non-actionable anomaly
                (False). Index must match the data type of `event_start` and `event_end`. Default: None.
            evaluate_until_event_end (str or bool): If True, evaluation is capped at event_end for all events.
                Allowed string values: 'normal_only', 'anomaly_only'. Default: False.
            event_id (int): ID of event. If not specified, a counter is used instead. Defaults to None.
            ignore_normal_index (bool): Whether to ignore the normal index and evaluate all data points in the prediction
                or test dataset. Default False.

        Returns:
            dict: Dictionary with computed metrics, e.g.:
                {
                    'event_id': int,
                    'event_label': str,
                    'weighted_score': float,
                    'max_criticality': float,
                    'f_beta_score': float or NaN,
                    'accuracy': float,
                    'tp': int, 'fp': int, 'tn': int, 'fn': int
                }

        Raises:
            ValueError: If event_label is invalid, evaluate_until_event_end has an unknown value,
                or if no data could be selected for the event.

        Notes:
            - The function sorts inputs by index to ensure alignment.
            - If normal_index is provided, this also influences the criticality calculation: criticality does not change
            if the expected behaviour is not normal.
            - If predicted_anomalies_event is empty, a ValueError is raised.
            - Use evaluate_until_event_end to control whether post-event predictions are considered.
        """

        if event_label not in ['anomaly', 'normal']:
            raise ValueError('Unknown event label (should be either `anomaly` or `normal`')

        if evaluate_until_event_end not in [True, False, 'normal_only', 'anomaly_only', 'True', 'False']:
            raise ValueError(f"Unknown value for `evaluate_until_event_end`. Should be on of: "
                             "{[True, False, 'normal_only', 'anomaly_only', 'True', 'False']}")

        # Sort input series to ensure they align
        predicted_anomalies = predicted_anomalies.sort_index()
        if normal_index is None or ignore_normal_index:
            normal_index = pd.Series([True] * len(predicted_anomalies), index=predicted_anomalies.index)
        else:
            normal_index = normal_index.sort_index()

        # determine data to evaluate
        cap_data = (
            (evaluate_until_event_end in [True, 'True'])
            | (event_label == 'anomaly' and evaluate_until_event_end == 'anomaly_only')
            | (event_label == 'normal' and evaluate_until_event_end == 'normal_only')
        )

        if cap_data:
            predicted_anomalies = predicted_anomalies.loc[:event_end]
            normal_index = normal_index.loc[:event_end]

        # Create the ground truth based on event start and end
        ground_truth = self.create_ground_truth(event_start, event_end, normal_index, event_label)

        # Select data to evaluate
        predicted_anomalies_event = predicted_anomalies.loc[event_start:event_end]
        if predicted_anomalies_event.empty:
            raise ValueError('No event data could be selected.')

        # Calculate metrics
        if event_label == 'anomaly':
            weighted_score_value = self._calculate_weighted_score(event_prediction=predicted_anomalies_event)
        else:
            weighted_score_value = np.nan

        # Calculate max criticality up till fault start (=event end)
        max_criticality = np.max(
            calculate_criticality(anomalies=predicted_anomalies.loc[:event_end],
                                  normal_idx=normal_index.loc[:event_end])
        )

        # We only evaluate predicted anomalies during expected normal operation
        metrics = self._calculate_basic_metrics(
            ground_truth=ground_truth[normal_index],  # What if an event only has anomaly as ground truth label?
            predicted_anomalies=predicted_anomalies[normal_index],
            event_label=event_label
        )

        if event_id is None:
            event_id = len(self._evaluated_events)

        evaluation = {
            'event_id': event_id,
            'event_label': event_label,  # true label
            'weighted_score': weighted_score_value,
            'max_criticality': max_criticality,
            **metrics  # F-score, Accuracy, TP, TN, FN, FP
        }

        # Add evaluation to dataframe
        self._evaluated_events.append(evaluation)

        return evaluation

    def get_final_score(self, event_selection: Optional[List[int]] = None, criticality_threshold: Optional[int] = None,
                        min_fraction_anomalous: Optional[float] = None) -> float:
        """Calculate the CARE-score for selected evaluated events.

        The CARE score combines average Coverage (pointwise F-score for anomaly events), average Earliness (weighted
        score for anomaly events), average Accuracy (for normal events) and Reliability (eventwise F-score) using the
        configured weights.

        If the average accuracy over all normal events < 0.5, CARE-score = average accuracy over all normal events
            (worse than random guessing).
        If no anomalies were detected, the CARE-score = 0.
        Else, the CARE-score is calculated as:

            ( (average F-score over all anomaly events) * coverage_w
             + (average weighted score over all anomaly events) * weighted_score_w
             + (average accuracy over all normal events) * accuracy_w
             + event wise F-score * eventwise_f_score_w ) / sum_of_weights

        where `sum_of_weights` = coverage_w + weighted_score_w + accuracy_w + eventwise_f_score_w.

        Args:
            event_selection (List[int]): list of event IDs to include. Default: None (use all).
            criticality_threshold (int):  If provided and anomaly_detection_method == 'criticality', override the stored
                threshold for this calculation.
            min_fraction_anomalous (float): If provided and anomaly_detection_method == 'fraction',  override the stored
                min_fraction_anomalous for this calculation.

        Returns:
            float: CARE-score

        Raises:
            ValueError: If the selected events do not contain at least one normal and one anomalous event.
        """

        # Reset threshold if necessary
        if self.anomaly_detection_method == 'criticality' and criticality_threshold:
            self.criticality_threshold = criticality_threshold
        if self.anomaly_detection_method == 'fraction' and min_fraction_anomalous:
            self.min_fraction_anomalous = min_fraction_anomalous

        events_to_evaluate = self._select_events(event_selection)

        if events_to_evaluate['event_label'].nunique() < 2:
            raise ValueError('To calculate the CARE Score, we need at least 1 normal event and 1 anomalous event.')

        if np.sum(events_to_evaluate['anomaly_detected']) == 0:
            logger.info('No anomalies were detected')
            return 0.

        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'

        avg_accuracy = events_to_evaluate.loc[~is_anomaly_event, 'accuracy'].mean()
        if avg_accuracy <= 0.5:
            logger.info('Accuracy over all normal events <0.5')
            return avg_accuracy

        avg_f_score = events_to_evaluate.loc[is_anomaly_event, 'f_beta_score'].mean()
        avg_weighted_score = events_to_evaluate.loc[is_anomaly_event, 'weighted_score'].mean()
        eventwise_fscore = self.calculate_reliability(event_selection)

        care_score = (avg_accuracy * self.accuracy_w
                      + avg_weighted_score * self.earliness_w
                      + avg_f_score * self.coverage_w
                      + eventwise_fscore * self.reliability_w)
        sum_of_weights = (self.accuracy_w + self.earliness_w +
                          self.coverage_w + self.reliability_w)
        care_score /= sum_of_weights

        return care_score

    def calculate_avg_coverage(self, event_selection: Optional[List[int]] = None) -> float:
        """Return the average Coverage (pointwise F-score) for anomaly events.

        Args:
            event_selection (list[int], optional): List of event IDs to include.
                Default: None (use all evaluated events).

        Returns:
            float: Mean f_beta_score for selected anomaly events. Returns numpy.nan if no anomaly events are selected.
        """
        events_to_evaluate = self._select_events(event_selection)
        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'
        return events_to_evaluate.loc[is_anomaly_event, 'f_beta_score'].mean()

    def calculate_avg_accuracy(self, event_selection: Optional[List[int]] = None) -> float:
        """Return the average Accuracy across normal events.

        Args:
            event_selection (list[int], optional): List of event IDs to include.
                Default: None (use all evaluated events).

        Returns:
            float: Mean accuracy for selected normal events. Returns numpy.nan if no normal events are selected.
        """
        events_to_evaluate = self._select_events(event_selection)
        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'
        return events_to_evaluate.loc[~is_anomaly_event, 'accuracy'].mean()

    def calculate_reliability(self, event_selection: Optional[List[int]] = None, **kwargs) -> float:
        """Compute the Reliability (event-wise F-score) for selected events.

        Args:
            event_selection (list[int], optional): List of event IDs to include.
                Default: None (use all evaluated events).
            kwargs: Other keyword args for sklearn's fbeta_score.

        Returns:
            float: Event-wise F-score computed with beta=self.reliability_beta.
                   If there are no positive labels or predictions, sklearn's fbeta_score behavior
                   is controlled by zero_division.
        """
        events_to_evaluate = self._select_events(event_selection)

        return fbeta_score(
            y_true=events_to_evaluate['event_label'] == 'anomaly',
            y_pred=events_to_evaluate['anomaly_detected'],
            beta=self.reliability_beta,
            **kwargs
        )

    def calculate_avg_earliness(self, event_selection: Optional[List[int]] = None) -> float:
        """Return the average Earliness (weighted score) for anomaly events.

        Args:
            event_selection (list[int], optional): List of event IDs to include.
                Default: None (use all evaluated events).

        Returns:
            float: Mean weighted_score for selected anomaly events. Returns numpy.nan if no anomaly events are selected.
        """
        events_to_evaluate = self._select_events(event_selection)
        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'
        return events_to_evaluate.loc[is_anomaly_event, 'weighted_score'].mean()

    @staticmethod
    def create_ground_truth(event_start: Union[int, pd.Timestamp], event_end: Union[int, pd.Timestamp],
                            normal_index: pd.Series, event_label: str) -> pd.Series:
        """Create the ground truth labels based on the event_start, event_end, normal_index and event_label.

        Args:
            event_start (int, pd.Timestamp): Start index/timestamp of the event.
            event_end (int, pd.Timestamp): End index/timestamp of the event.
            normal_index (pd.Series): Boolean mask indicating normal samples. Must be indexed compatibly with
                event_start/event_end.
            event_label (str): 'anomaly' or 'normal'. True label indicating the type of the event.

        Returns:
            pd.Series: Boolean series indexed like normal_index. True indicates anomaly (actionable and non-actionable),
                False otherwise.

        Notes:
            - The returned series is sorted by index.
            - If event_label == 'anomaly', values in the interval [event_start:event_end] are set to True.
            - normal_index is inverted to start (anomalies = not normal), then the event window is applied.
        """

        ground_truth = pd.Series(data=~normal_index, index=normal_index.index, name='ground_truth')
        ground_truth = ground_truth.sort_index()
        if event_label == 'anomaly':
            ground_truth.loc[event_start:event_end] = True
        return ground_truth

    def _calculate_basic_metrics(self, ground_truth: pd.Series, predicted_anomalies: pd.Series,
                                 event_label: str) -> Dict[str, float]:
        """Calculates F-Score, accuracy and the confusion matrix counts for the given labels and predictions.

        Args:
            ground_truth (pd.Series): Ground truth labels (False = normal behaviour).
            predicted_anomalies (pd.Series): Boolean predictions (True = anomaly detected).
            event_label (str): 'anomaly' or 'normal'. Determines whether f_beta is computed.

        Returns:
            Dict[str, float]: {
                'f_beta_score': float (or np.nan if not applicable),
                'accuracy': float,
                'tn': int, 'fp': int, 'fn': int, 'tp': int
            }

        Notes:
            - If event_label == 'anomaly', f_beta_score is computed using self.coverage_beta.
            - If ground_truth contains all True values, a UserWarning is emitted because F-score may be degenerate.
        """

        f_score = np.nan
        if event_label == 'anomaly':
            # When only the event is evaluated, the precision is either 0 (none of the data points detected
            # or 1 (at least 1 point detected as anomaly). The recall is the same as accuracy.
            if all(ground_truth):
                warnings.warn('UserWarning: Ill defined anomaly-event, ground truth contains only True values. This can'
                              'lead to degenerated F-Score values.')
            f_score = fbeta_score(y_true=ground_truth, y_pred=predicted_anomalies, beta=self.coverage_beta)

        accuracy = accuracy_score(y_true=ground_truth, y_pred=predicted_anomalies)
        tn, fp, fn, tp = confusion_matrix(y_true=ground_truth, y_pred=predicted_anomalies, labels=[False, True]).ravel()

        return {
            'f_beta_score': f_score,
            'accuracy': accuracy,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        }

    def _calculate_weighted_score(self, event_prediction: pd.Series) -> float:
        """Calculate the weighted score (earliness) for a single event prediction series.

        For each element of event_prediction, this function computes a weight between 0 and 1 which is then multiplied
        with the anomaly prediction of the element. In the end, the weighted score is the normalized sum of weights *
        event_prediction.

        Args:
            event_prediction (pd.Series): Boolean series for the event (sorted chronologically).
                True = anomaly detected, False = normal.

        Returns:
            float: Weighted score for the event; higher means earlier detection.

        Raises:
            ValueError: If ws_start_of_descend is invalid (denominator <= 0 or numerator/denominator >= 1).
        """

        event_length = len(event_prediction)

        numerator, denominator = self.ws_start_of_descend
        if denominator <= 0:
            raise ValueError("ws_start_of_descend denominator must be > 0")
        start_of_descend = numerator / denominator
        if not (0 <= start_of_descend < 1):
            raise ValueError("ws_start_of_descend must satisfy 0 <= numerator/denominator < 1")

        scale = int(denominator)
        scaled_event_length = event_length * scale
        cp = int(scaled_event_length * start_of_descend)

        x_values = np.linspace(0, scale, scaled_event_length)
        weights = np.zeros(scaled_event_length)
        weights[:cp] = scale
        slope = 1 / (1 - start_of_descend)
        offset = scale / (1 - start_of_descend)
        weights[cp:scaled_event_length] = offset - slope * x_values[cp:scaled_event_length]

        final_weights = weights[::scale]
        weights = final_weights / scale

        return np.sum(event_prediction * weights) / np.sum(weights)

    def save_evaluated_events(self, file_path: Union[Path, str]) -> None:
        """Write the evaluated events to a CSV file.

        Args:
            file_path (Path or str): The file path where the evaluated events will be saved.
        """
        self.evaluated_events.to_csv(Path(file_path), index=False)

    def load_evaluated_events(self, file_path: Union[Path, str]) -> None:
        """Load evaluated events from a CSV file and replace the internal evaluated-events list.

        Args:
            file_path (Path or str): The file path from which the evaluated events will be loaded.
        """
        file_path = Path(file_path)
        try:
            self._evaluated_events = pd.read_csv(file_path).to_dict(orient='records')
        except Exception as exc:
            raise ValueError(f"Failed to read evaluated events from {file_path}: {exc}") from exc

    def _select_events(self, event_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Return a DataFrame of selected evaluated events. The returned DataFrame is a view/copy of the internal data
        (constructed via the evaluated_events property)."""

        if event_ids is None:
            return self.evaluated_events

        return self.evaluated_events[self.evaluated_events['event_id'].isin(event_ids)]
