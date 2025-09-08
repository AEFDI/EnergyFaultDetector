
from typing import Optional, Dict, Tuple, List, Any, Union
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix

from energy_fault_detector.utils.analysis import calculate_criticality

logger = logging.getLogger('energy_fault_detector')


class CAREScore:
    """Calculate CARE Score (coverage, accuracy, reliability, and earliness) for an anomaly detection algorithm, as
    described in the paper 'CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine
    Data' (https://doi.org/10.3390/data9120138).

    For each normal and anomalous event in your dataset call `evaluate_event`. Afterward, call `get_final_score`
    to calculate the CARE score of your model.

    Coverage is measured by the pointwise F-Score of anomalous events, the accuracy measured as the accuracy of normal
    events, the reliability by the eventwise F-score over all events and the earliness as weighted score over anomalous
    events.

    The method `evaluate_event` calculates the coverage, accuracy and weighted score for a specific event, as well as
    whether the event was detected as anomaly or not. It collects this information in a dataframe, which can be
    accessed through the `evaluated_events` property.

    Finally, the `get_final_score` is called to calculate the CARE score over all (or a part of the) evaluated events.

    Args:
        coverage_beta (float): Beta parameter for coverage F-score calculation. Default is 0.5.
        eventwise_f_score_beta (float): Beta parameter for event-wise F-score calculation. Default is 0.5.
        coverage_w (float): Weight for coverage in the final CARE score calculation. Default is 1.0.
        accuracy_w (float): Weight for accuracy in the final CARE score calculation. Default is 2.0.
        weighted_score_w (float): Weight for the weighted score in the final CARE score calculation. Default is 1.0.
        eventwise_f_score_w (float): Weight for the event-wise F-score in the final CARE score calculation.
            Default is 1.0.
        anomaly_detection_method (str): Method used to calculate anomaly detection score. Either criticality or
            fraction. Default is 'criticality'. If fraction is used, an event is detected as anomaly if at least
            `min_fraction_anomalous_timestamps` of the data points are detected as anomaly. If criticality is
            used, an event is detected as anomaly if the maximum criticality exceed criticality_threshold.
        criticality_threshold (int): Threshold for criticality. If criticality exceeds this threshold, the event will be
            detected as an anomaly, if `anomaly_detection_method` == 'criticality'. Default is 72.
        min_fraction_anomalous (float): Minimum fraction of anomalous data points to consider an event
            detected anomaly, if `anomaly_detection_method` == 'fraction'. Default is 0.1.
        ws_start_of_descend (Tuple[int, int]): Must be a fraction (tuple with numerator and denominator) between 0
            and 1. It determines the point after which the scoring weights for the weighted score decrease.
            Default is (1, 4).
        min_fraction_anomalous_timestamps: deprecated, use min_fraction_anomalous instead.

    """

    def __init__(self, coverage_beta: float = 0.5, eventwise_f_score_beta: float = 0.5, coverage_w: float = 1.,
                 accuracy_w: float = 2., weighted_score_w: float = 1., eventwise_f_score_w: float = 1.,
                 criticality_threshold: int = 72, min_fraction_anomalous: float = 0.1,
                 ws_start_of_descend: Tuple[int, int] = (1, 4), anomaly_detection_method: str = 'criticality',
                 min_fraction_anomalous_timestamps: float = None):

        if min_fraction_anomalous_timestamps is not None:
            warnings.warn(
                '`min_fraction_anomalous_timestamps`is deprecated and will be removed in future versions. '
                'Please use `min_fraction_anomalous` instead.',
                DeprecationWarning,
                stacklevel=2
            )
            min_fraction_anomalous = min_fraction_anomalous_timestamps

        if anomaly_detection_method not in ['criticality', 'fraction']:
            raise ValueError("Anomaly detection method must be either 'criticality' or 'fraction'")

        self.coverage_beta = coverage_beta
        self.eventwise_f_score_beta = eventwise_f_score_beta

        self.coverage_w = coverage_w
        self.accuracy_w = accuracy_w
        self.weighted_score_w = weighted_score_w
        self.eventwise_f_score_w = eventwise_f_score_w

        self.min_fraction_anomalous = min_fraction_anomalous
        self.criticality_threshold = criticality_threshold
        self.anomaly_detection_method = anomaly_detection_method

        self.ws_start_of_descend = ws_start_of_descend

        self._evaluated_events: List[Dict[str, Any]] = []

    @property
    def evaluated_events(self) -> pd.DataFrame:
        """Pandas DataFrame with evaluated events."""
        df = pd.DataFrame(self._evaluated_events)

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
        """Evaluate the performance of a fault detection model for a given event.

        If a `normal_index` is provided, the metrics are only calculated for data points where we expected normal
        behaviour. The argument `evaluate_until_event_end` determines which part of the provided data is used for
        evaluation. It might be useful to set this to True or `anomaly_only` if you expect normal behaviour may change
        after a fault, in which case the model will predict many false positives after `event_end`, reducing the final
        score.

        Args:
            event_start (int, pd.Timestamp): Start index/timestamp of the event.
            event_end (int, pd.Timestamp): End index/timestamp of the event.
            event_label (str): True label of the event. This can be either 'anomaly' or 'normal'.
            predicted_anomalies (pd.Series): Boolean pandas series, indicating whether an anomaly was detected.
                Index must match the data type of `event_start` and `event_end`.
            normal_index (pd.Series, optional): Boolean mask indicating normal behaviour. Not used if not provided.
                Index must match the data type of `event_start` and `event_end`. Defaults to None.

                This mask helps to identify the data points which are interesting to evaluate. If there are data
                points which are easy to detect as anomaly, because the operational status of the device or asset is
                not 'normal', these data points should be ignored to evaluate the fault detection model properly.
                These timestamps are also taken into account when calculating the criticality. The criticality does not
                increase or decrease when no normal behaviour is expected, i.e. when `normal_index == False`.
                Note that you should not mark a complete anomaly event as False, since this would essentially remove
                these data points from the evaluation. Only mark data points als False, if the operational status is
                not normal behaviour, e.g. under maintenance, idling, fault active, etc.
            evaluate_until_event_end (str or bool): Indicates whether to evaluate all data points in the provided test
                (`predicted_anomalies`) or only up until the `event_end`, which is the start of a fault in case of an
                anomalous event (`event_label == 'anomaly'`). Allowed values are True, False, normal_only, anomaly_only.
                Defaults to False.

                Note that it is useful to set this to True or `anomaly_only` if you expect normal behaviour may change
                after a fault, in which case the model will predict many false positives after `event_end`.
                True will cap the evaluation data of all events, 'normal_only' only for normal events, 'anomaly_only'
                only for anomalous events and False evaluates the data as is, including any data after `event_end`.
            event_id (int): ID of event. If not specified, a counter is used instead. Defaults to None.
            ignore_normal_index (bool): Whether to ignore the normal index and evaluate all data points in the prediction
                or test dataset. Default False.

        Returns:
            Dict[str, [int or float]]: Dictionary containing the calculated metrics.
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
        metrics = self._calculate_metrics(
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
        """Calculate the CARE score over all events in self.evaluated_events or a selection of the events.

        If the average accuracy over all normal events < 0.5, CARE score = average accuracy over all normal events
            (worse than random guessing).
        If no anomalies were detected, the CARE score = 0.
        Else, the CARE score is calculated as:

            ( (average F-score over all anomaly events) * coverage_w
             + (average weighted score over all anomaly events) * weighted_score_w
             + (average accuracy over all normal events) * accuracy_w
             + event wise F-score * eventwise_f_score_w ) / sum_of_weights

        where `sum_of_weights` = coverage_w + weighted_score_w + accuracy_w + eventwise_f_score_w.

        Args:
            event_selection: list of event IDs to calculate the CARE score for.
            criticality_threshold: Reset the criticality threshold, if `anomaly_detection_method` == 'criticality'.
                Useful to test different alert thresholds.
            min_fraction_anomalous: Reset the minimum fraction of anomalies, `anomaly_detection_method` == 'fraction'.
                Useful to test different alert thresholds.

        Returns:
            float: CARE score
        """

        if self.anomaly_detection_method == 'criticality' and criticality_threshold:
            self.criticality_threshold = criticality_threshold
        if self.anomaly_detection_method == 'fraction' and min_fraction_anomalous:
            self.min_fraction_anomalous = min_fraction_anomalous

        events_to_evaluate = self._select_events(event_selection)

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
        eventwise_fscore = self.calculate_eventwise_f_score(event_selection)

        care_score = (avg_accuracy * self.accuracy_w
                      + avg_weighted_score * self.weighted_score_w
                      + avg_f_score * self.coverage_w
                      + eventwise_fscore * self.eventwise_f_score_w)
        sum_of_weights = (self.accuracy_w + self.weighted_score_w +
                          self.coverage_w + self.eventwise_f_score_w)
        care_score /= sum_of_weights

        return care_score

    def calculate_avg_accuracy(self, event_selection: Optional[List[int]] = None) -> float:
        """Calculate the avg accuracy across all normal events or selected normal events.

        Args:
            event_selection: list of event IDs to calculate the score for.

        Returns:
            Average accuracy for normal events.
        """
        events_to_evaluate = self._select_events(event_selection)
        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'
        return events_to_evaluate.loc[~is_anomaly_event, 'accuracy'].mean()

    def calculate_avg_weighted_score(self, event_selection: Optional[List[int]] = None) -> float:
        """Calculate average weighted score (earliness score) for all anomaly events or selected anomaly events.

        Args:
            event_selection: list of event IDs to calculate the score for.

        Returns:
            Average weighted score for all anomaly events.
        """
        events_to_evaluate = self._select_events(event_selection)
        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'
        return events_to_evaluate.loc[is_anomaly_event, 'weighted_score'].mean()

    def calculate_avg_f_score(self, event_selection: Optional[List[int]] = None) -> float:
        """Calculate average F-score (coverage score) for all anomaly events or selected anomaly events.

        Args:
            event_selection: list of event IDs to calculate the score for.

        Returns:
            Average weighted score for all anomaly events.
        """
        events_to_evaluate = self._select_events(event_selection)
        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'
        return events_to_evaluate.loc[is_anomaly_event, 'f_beta_score'].mean()

    def calculate_eventwise_f_score(self, event_selection: Optional[List[int]] = None) -> float:
        """Calculate the eventwise F-Score (reliability score) across all events or selected events.

        Args:
            event_selection: list of event IDs to calculate the score for.

        Returns:
            Eventwise F-Score.
        """
        events_to_evaluate = self._select_events(event_selection)

        return fbeta_score(
            y_true=events_to_evaluate['event_label'] == 'anomaly',
            y_pred=events_to_evaluate['anomaly_detected'],
            beta=self.eventwise_f_score_beta
        )

    @staticmethod
    def create_ground_truth(event_start: Union[int, pd.Timestamp], event_end: Union[int, pd.Timestamp],
                            normal_index: pd.Series, event_label: str) -> pd.Series:
        """Create the ground truth labels based on the provided inputs.

        Args:
            event_start (int, pd.Timestamp): Start index/timestamp of the event.
            event_end (int, pd.Timestamp): End index/timestamp of the event.
            normal_index (pd.Series): Boolean mask indicating normal samples.
            event_label (str): True label indicating the type of the event (anomaly or normal).

        Returns:
            pd.Series: Ground truth labels. True if the timestamp is an anomaly, False otherwise.
        """

        ground_truth = pd.Series(data=~normal_index, index=normal_index.index, name='ground_truth')
        ground_truth = ground_truth.sort_index()
        if event_label == 'anomaly':
            ground_truth.loc[event_start:event_end] = True
        return ground_truth

    def _calculate_metrics(self, ground_truth: pd.Series, predicted_anomalies: pd.Series,
                           event_label: str) -> Dict[str, float]:
        """Calculate various metrics for the given labels and scores.

        Args:
            ground_truth (pd.Series): Ground truth labels.
            predicted_anomalies (pd.Series): Predicted labels.
            event_label (str): normal or anomaly, indicates whether F-beta-Score can be calculated.

        Returns:
            Dict[str, float]: Dictionary containing the calculated metrics.
        """

        f_score = np.nan
        if event_label == 'anomaly':
            # When only the event is evaluated, the precision is either 0 (none of the data points detected
            # or 1 (at least 1 point detected as anomaly). The recall is the same as accuracy.
            # TODO: UserWarning
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
        """Calculate the weighted score based on a modification of the linear weighting function.

        For each element of event_prediction, this function computes a weight between 0 and 1 which is then multiplied
        with the anomaly prediction of the element. In the end, the weighted score is the normalized sum of weights *
        event_prediction.

        Args:
            event_prediction (pd.Series): Boolean model prediction during the event. True = Anomaly and False = Normal.
                The series must be sorted chronologically!

        Returns:
            float: Weighted score for the event; higher means earlier detection.
        """

        event_length = len(event_prediction)

        start_of_descend_numerator, start_of_descend_denominator = self.ws_start_of_descend
        start_of_descend_float = start_of_descend_numerator / start_of_descend_denominator

        scale = start_of_descend_denominator
        scaled_event_length = event_length * scale
        cp = int(scaled_event_length * start_of_descend_float)

        x_values = np.linspace(0, scale, scaled_event_length)
        weights = np.zeros(scaled_event_length)
        weights[:cp] = scale
        slope = 1 / (1 - start_of_descend_float)
        offset = scale / (1 - start_of_descend_float)
        weights[cp:scaled_event_length] = offset - slope * x_values[cp:scaled_event_length]

        final_weights = weights[::scale]
        weights = final_weights / scale

        return np.sum(event_prediction * weights) / np.sum(weights)

    def save_evaluated_events(self, file_path: Union[Path, str]) -> None:
        """Save the evaluated events to a CSV file.

        Args:
            file_path (Path): The file path where the evaluated events will be saved.
        """
        self.evaluated_events.to_csv(Path(file_path), index=False)

    def load_evaluated_events(self, file_path: Union[Path, str]) -> None:
        """Load evaluated events from a CSV file.

        Args:
            file_path (Path): The file path from which the evaluated events will be loaded.
        """
        file_path = Path(file_path)
        if file_path.exists():
            self._evaluated_events = pd.read_csv(file_path).to_dict(orient='records')
        else:
            raise FileNotFoundError(f"File {file_path} does not exist.")

    def _select_events(self, event_ids: Optional[List[int]] = None) -> pd.DataFrame:
        if event_ids is None:
            return self.evaluated_events

        return self.evaluated_events[self.evaluated_events['event_id'].isin(event_ids)]
