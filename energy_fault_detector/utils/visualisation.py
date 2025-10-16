"""Visualisation functions."""

from typing import Union, Tuple, Optional, List
import warnings

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pandas as pd

from energy_fault_detector.core import Autoencoder
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.utils.analysis import calculate_criticality


def plot_learning_curve(model: Union[Autoencoder, FaultDetector], ax: plt.Axes = None, label: str = '',
                        **subplot_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the learning curve of the specified model.

    Args:
        model (Union[Autoencoder, FaultDetector]): The model for which to plot the learning curve.
        ax (Optional[plt.Axes], optional): Axes to plot the learning curve on. Defaults to None.
        label (str, optional): Label for the learning curve. Defaults to ''.
        subplot_kwargs (dict, optional): Additional keyword arguments for subplots, if no axes are passed.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(**subplot_kwargs)
    else:
        fig = ax.get_figure()

    if isinstance(model, FaultDetector):
        model = model.autoencoder

    if model is None:
        raise ValueError('The model is not fitted.')

    if label != '':
        label = f'({label})'
    ax.plot(model.history['loss'], label=f'Train loss {label}')
    if 'val_loss' in model.history.keys():
        ax.plot(model.history['val_loss'], label=f'Validation loss {label}')
    ax.legend()
    return fig, ax


def plot_reconstruction(data: pd.DataFrame, reconstruction: pd.DataFrame, features_to_plot: Optional[List[str]] = None,
                        height_multiplier: float = 1.5, original_scale: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the original dataset and its reconstruction.

    Note: can result in a very large plot, if the dataset contains many columns/features. Use the
    `features_to_plot` parameter to specify which columns to plot.

    Args:
        data (pd.DataFrame): DataFrame containing the original data.
        reconstruction (pd.DataFrame): DataFrame containing the reconstructed data.
        features_to_plot (Optional[List[str]], optional): List of features to plot. Defaults to None (all features).
        height_multiplier (float, optional): Multiplier for the vertical size of the figure. Defaults to 1.5.
        original_scale (bool, optional): Whether to scale the y-axis using the input data. Defaults to True.
            If true, the y limits are set to minimum - std, maximum + std for each feature plotted.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes containing the plots, for further customization if needed.
    """
    to_plot = reconstruction.columns if features_to_plot is None else features_to_plot
    if any(col not in data.columns for col in to_plot):
        missing = set(to_plot) - set(data.columns)
        raise ValueError(f'The columns {missing} are not present in the dataset.')

    if len(to_plot) > 30:  # You can adjust this threshold
        warnings.warn(f"You are attempting to plot a large number of features ({len(to_plot)}). "
                      "This may result in a cluttered figure. Consider selecting fewer features to plot.")

    fig, ax = plt.subplots(len(to_plot), 1, figsize=(12, len(to_plot) * height_multiplier))

    # Plot each feature
    axes = [ax] if len(to_plot) == 1 else ax.flatten()
    for col, ax_ in zip(to_plot, axes):
        ax_.plot(data[col], label=col, alpha=0.5)
        if col in reconstruction.columns:
            ax_.plot(reconstruction[col], label='reconstruction', alpha=0.5)
        else:
            warnings.warn(f'No reconstruction available for column {col}.')
        ax_.legend(loc='upper left')
        if original_scale:
            ax_.set_ylim(data[col].min() - data[col].std(), data[col].max() + data[col].std())

    plt.tight_layout()
    return fig, ax


def plot_score_with_threshold(model: FaultDetector, data: pd.DataFrame, normal_index: pd.Series = None,
                              ax: plt.Axes = None, figsize: Tuple[float, float] = (8, 3),
                              show_predicted_anomaly: bool = False, show_threshold: bool = True,
                              show_criticality: bool = False, max_criticality: int = 144,
                              score_color: Optional[str] = None,
                              anomaly_color: Optional[str] = None,
                              criticality_color: Optional[str] = "C2",
                              threshold_color: Optional[str] = 'k', **subplot_kwargs
                              ) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the anomaly scores of the AnomalyDetector model along with the threshold for the provided data.

    Args:
        model (FaultDetector): The anomaly detection model used to compute the scores.
        data (pd.DataFrame): DataFrame containing the data for which scores are computed.
        normal_index (pd.Series): Boolean series indicating whether the data points have a normal status or not.
        ax (Optional[plt.Axes], optional): Axes object to plot on. If None, a new figure and axes will be created.
        figsize (Tuple[float, float], optional): Size of the figure if a new one is created. Defaults to (8, 3).
        show_predicted_anomaly (bool, optional): Whether to show the predicted anomaly scores. Defaults to False.
        show_threshold (bool, optional): Whether to show the threshold scores. Defaults to True.
        show_criticality (bool, optional): Whether to show the criticality counter. Defaults to False.
        max_criticality (int optional): If show_criticality is True, the maximum value of the criticality counter can
            be specified. Defaults to 144 (one day of 10 min timestamps).
        score_color (Optional[str], optional): Color to use for the anomaly score.
        anomaly_color (Optional[str], optional): Color to use for the anomalous data points (using normal_index).
        criticality_color (Optional[str], optional): Color to use for the criticality counter if show_criticality is
            True. Defaults to 'C2'.
        threshold_color (Optional[str], optional): Color to use for the threshold.
        **subplot_kwargs: Additional keyword arguments for plt.subplots().

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes containing the plot for further customization if needed.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **subplot_kwargs)
    else:
        fig = ax.get_figure()

    predictions = model.predict(data)
    scores = predictions.anomaly_score

    if normal_index is None and not show_predicted_anomaly:
        ax.scatter(scores.index, scores, s=1, alpha=0.8, c=score_color)
    elif show_predicted_anomaly:
        predicted_anomalies = predictions.predicted_anomalies
        ax.scatter(scores.index[~predicted_anomalies], scores[~predicted_anomalies], s=1, alpha=0.8, c=score_color)
        ax.scatter(scores.index[predicted_anomalies], scores[predicted_anomalies], s=1, alpha=0.8, c=anomaly_color,
                   label='predicted anomaly')
    elif normal_index is not None:
        ax.scatter(scores.loc[normal_index].index, scores.loc[normal_index], s=1, alpha=0.8, label='normal status',
                   c=score_color)
        ax.scatter(scores.loc[~normal_index].index, scores.loc[~normal_index], s=1, alpha=0.8, label='anomalous status',
                   c=anomaly_color)

    if show_threshold:
        if isinstance(model.threshold_selector.threshold, float):
            ax.axhline(model.threshold_selector.threshold, linestyle='--', label='threshold', c=threshold_color)
        else:
            threshold = model.threshold_selector.threshold
            if isinstance(scores, pd.Series):
                threshold = pd.Series(model.threshold_selector.threshold, index=scores.index)
            ax.plot(threshold, linestyle='-', linewidth=.7, label='threshold', c=threshold_color)

    if show_criticality:
        crit = calculate_criticality(predictions.predicted_anomalies["anomaly"], normal_idx=normal_index,
                                     max_criticality=max_criticality)
        ax2 = ax.twinx()
        ax2.plot(crit, label='criticality counter', color=criticality_color)
        ax2.legend(loc='upper right', markerscale=3)
        ax2.set_ylabel('criticality')

    ax.set_ylabel('anomaly score')

    legend = ax.legend(loc='upper left', markerscale=3)
    for h in legend.legend_handles:
        h.set_alpha(1)

    return fig, ax


def plot_arcana_mean_importances(importances: pd.Series, top_n_features: int = 10, figsize: Tuple = (8, 8),
                                 ax: plt.Axes = None, **subplot_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the ARCANA importances as a horizontal bar plot.

    Args:
        importances (pd.Series): Series containing the ARCANA importances of the features.
        top_n_features (int, optional): Number of features to plot. Defaults to 10.
        figsize (Tuple[float, float], optional): Size of the figure if a new one is created. Defaults to (8, 8).
        ax (Optional[plt.Axes], optional): Axes object to plot on. If None, a new figure and axes will be created.
        **subplot_kwargs: Additional keyword arguments for plt.subplots().

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes containing the plot for further customization if needed.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **subplot_kwargs)
    else:
        fig = ax.get_figure()
    to_plot = importances.sort_values(ascending=True)[-top_n_features:]
    ax.barh(y=to_plot.index,
            width=to_plot.values, align='center')
    ax.set_yticks(range(len(to_plot)))
    ax.set_yticklabels(to_plot.index, rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel('ARCANA importance')
    return fig, ax


def plot_arcana_losses(losses: pd.DataFrame, figsize: Tuple = (8, 8)) -> None:
    """Plots the graphs for the ARCANA losses: Loss 1, Loss 2, and Combined Loss.

    Args:
        losses (pd.DataFrame): A DataFrame with iteration numbers as index and the columns representing the losses.
        figsize (Tuple[float, float], optional): Size of the figure if a new one is created. Defaults to (8, 8).
    """
    # Create subplots for each loss type
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)  # 3 rows, 1 column
    fig.suptitle('Arcana Losses')

    # Loop through each Arcana version and plot each loss type
    for i, loss_name in enumerate(losses.columns):
        loss_values = losses[loss_name]
        iterations = list(losses.index)

        # Plot each loss type in the corresponding subplot
        axs[i].plot(iterations, loss_values, label=loss_name, linestyle='-', linewidth=2)
        axs[i].set_ylabel(loss_name)
        axs[i].set_xlabel('iterations')


def animate_bias(bias_list: List[pd.DataFrame], selected_column_names: List[str], filename: str = 'arcana_bias.gif',
                 figsize: Tuple = (8, 8)):
    """Plots biases as a bar plot and animates it over the iterations.

    Args:
        bias_list (List[pd.DataFrame]): A list of pandas DataFrames containing ARCANA biases recorded over the
            iterations.
        selected_column_names (List[str]): Names of the features which should be plotted.
        filename (str, optional): Name of the GIF file. Defaults to 'arcana_bias.gif'.
        figsize (Tuple[float, float], optional): Size of the figure if a new one is created. Defaults to (8, 8).
    """
    df_list = [x[selected_column_names] for x in bias_list]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(selected_column_names, np.abs(df_list[0]))
    ax.tick_params("x", rotation=45)

    def update_frame(frame):
        plt.title(f'Iteration {frame * 50}')
        for bar, height in zip(bars, np.abs(df_list[frame])):
            bar.set_height(height)
        return bars

    ax.set_ylim(0, np.max([np.max(np.abs(arr)) for arr in df_list]) + 1)
    ani = animation.FuncAnimation(fig, update_frame, interval=200,  frames=len(df_list), repeat=False)
    ani.save(filename, writer='pillow')


def animate_corrected_input(corrected_list: List[pd.DataFrame], selected_column_names: List[str],
                            initial_input: pd.DataFrame, expected_result: pd.DataFrame = None,
                            filename: str = 'arcana_corrected_input.gif', figsize: Tuple = (8, 8)):
    """Plots graphs of ARCANA corrected inputs, which are the initial_input + bias, and animates them.

    Args:
        corrected_list (List[pd.DataFrame]): A list of pandas DataFrames containing ARCANA corrected inputs recorded
            over the iterations.
        selected_column_names (List[str]): Names of the features which should be plotted.
        initial_input (pd.DataFrame): Input at the starting point of the ARCANA optimization.
        expected_result (Optional[pd.DataFrame], optional): Expected normal behavior for debugging and verifying
            results. Defaults to None.
        filename (str, optional): Name of the GIF file. Defaults to 'arcana_corrected_input.gif'.
        figsize (Tuple[float, float], optional): Size of the figure if a new one is created. Defaults to (8, 8).
    """
    fig, axs = plt.subplots(nrows=len(selected_column_names), figsize=figsize, sharex=True)
    df_list = [x[selected_column_names] for x in corrected_list]
    selected_initial_input = initial_input[selected_column_names]
    new_column_names = {}
    for name in selected_column_names:
        new_column_names[name] = 'original ' + name

    # determine y_max values for each subplot
    addon_list = [selected_initial_input]
    if expected_result is not None:
        selected_expected_result = expected_result[selected_column_names]
        addon_list.append(selected_expected_result)
        selected_expected_result = selected_expected_result.rename(columns=new_column_names)

    for i, column in enumerate(selected_column_names):
        y_max = np.max([np.max(df[column]) for df in df_list + addon_list])
        axs[i].set_ylim(0, y_max + 0.25)
        axs[i].tick_params("x", rotation=45)

    # Animation function
    def update_frame(frame):
        """Plots the initial input and the corrected input from the ARCANA iteration number determined by the current frame.
        Optionally, expected results can also be plotted if they are provided. """
        for ax in axs:
            ax.clear()
        for i, column in enumerate(selected_column_names):
            axs[i].plot(selected_initial_input[column], alpha=0.5)
            if expected_result is not None:
                axs[i].plot(selected_expected_result[column], alpha=0.5)
            df = df_list[frame]
            axs[i].plot(df[column])
            if len(column) > 15:
                axs[i].set_ylabel(column[:15] + "...",)
            else:
                axs[i].set_ylabel(column)
            axs[i].set_ylim(0, y_max + 0.25)
            fig.suptitle(f'Iteration {frame * 50}')
        return axs

    ani = animation.FuncAnimation(fig, update_frame, interval=200, frames=len(df_list), repeat=False)
    ani.save(filename, writer='pillow')


def plot_arcana_importance_series(importances: List[pd.DataFrame], num_features: Optional[int] = 5,
                                  anomaly_events: Optional[pd.DataFrame] = None,
                                  ax: Optional[plt.Axes] = None, figsize: Optional[Tuple[float, float]] = (8, 3),
                                  **subplot_kwargs):
    """Plots the importance time series for the features with the highest average importance across all timestamps.

    Args:
        importances (List[pd.DataFrame]): Contains pandas DataFrames with importance values for different features
            (columns) across a range of timestamps (index).
        num_features (Optional[int]): Number of features to plot for each event. The features with the highest maximum
            importance across all analyzed event timestamps are selected. Defaults to 5.
        anomaly_events (Optional[pd.DataFrame]): DataFrame with columns 'start', 'end', and 'duration' describing
            detected anomaly events. Defaults to None.
        figsize (Tuple[float, float], optional): Size of the figure if a new one is created. Defaults to (8, 8).
        ax (Optional[plt.Axes], optional): Axes object to plot on. If None, a new figure and axes will be created.
        **subplot_kwargs: Additional keyword arguments for plt.subplots().

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **subplot_kwargs)
    else:
        fig = ax.get_figure()

    top_feature_list = []
    for importance in importances:
        top_feature_list += list(importance.columns[importance.mean(axis=0).argsort()[-num_features:].values])
    top_features = set(top_feature_list)

    all_lines = []
    for i, feature in enumerate(top_features):
        feature_lines = tuple()
        for importance in importances:
            line = ax.plot(importance[feature], color=f'C{i}', marker='o')
            feature_lines += tuple(line)
        all_lines.append(feature_lines)
    ax.legend(handles=all_lines, labels=top_features, handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_yscale('log')
    if anomaly_events is not None:
        for i in range(len(anomaly_events)):
            ax.axvspan(anomaly_events.iloc[i]['start'], anomaly_events.iloc[i]['end'], alpha=0.1, color='red')
