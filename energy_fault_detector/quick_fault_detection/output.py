import os
from typing import Union, List, Optional

import pandas as pd
from matplotlib import pyplot as plt

from energy_fault_detector.fault_detector import FaultDetector
import energy_fault_detector.utils.visualisation as viz

output_info = """The QuickFaultDetector’s output may include features that were transformed during the 
anomaly-detection process. To avoid false anomalies, any given angle features are first converted into continuous 
representations via sine and cosine transformations. These transformed features can appear as 'feature_sine' and 
'feature_cosine' in the ARCANA Importance plots."""


def generate_output_plots(anomaly_detector: FaultDetector, train_data: pd.DataFrame,
                          normal_index: Union[pd.Series, None], test_data: pd.DataFrame, event_meta_data: pd.DataFrame,
                          arcana_mean_importances: List[pd.Series], arcana_losses: List[pd.DataFrame],
                          save_dir: Optional[str] = None) -> None:
    """ Generates output plots based on failure detection results. The default output presented in a subplot with
    2 rows and 2 columns containing:
    1. Prediction anomaly score plot with marked anomaly events + threshold
    2. Training anomaly score plot + threshold
    3. Learning curve plot of the autoencoder
    4. Optionally ARCANA-importances if anomaly events have been detected.
    Optional debug plots are provided if arcana_losses are provided.

    Args:
        anomaly_detector (FaultDetector): Trained AnomalyDetector instance
        train_data (pd.DataFrame): dataframe containing the numerical data used for the AnomalyDetector training.
        normal_index (Union[pd.Series, None]):
        test_data (pd.DataFrame): dataframe containing the data used for evaluation.
        event_meta_data (pd.Dataframe): Potentially empty dataframe containing information about event starts, ends and
            durations if there are anomaly events.
        arcana_mean_importances (List[pd.Series]): If anomalies are present this list contains a pandas series
            for each event which contains the mean Arcana-importance values for every feature in the data.
        arcana_losses (List[pd.DataFrame]): Potentially empty List of dataframe containing recorded ARCANA losses for
            each event if the losses were tracked.
        save_dir (Optional[str]): Directory to save the output plots. If not provided, the plots are not saved.
            Defaults to None.

    """
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs[0, 0].set_title('Anomaly Score of Prediction Data')
    viz.plot_score_with_threshold(model=anomaly_detector, data=test_data, normal_index=None, ax=axs[0, 0])
    axs[0, 0].set_yscale('log')
    for i in range(len(event_meta_data)):
        axs[0, 0].axvspan(event_meta_data.iloc[i]['start'],
                          event_meta_data.iloc[i]['end'], alpha=0.1, color='red')
    axs[0, 1].set_title('Anomaly Score During Training')
    viz.plot_score_with_threshold(model=anomaly_detector, data=train_data, normal_index=normal_index, ax=axs[0, 1])
    axs[0, 1].set_yscale('log')

    viz.plot_learning_curve(anomaly_detector, ax=axs[1, 0])
    axs[1, 0].set_title('Model Learning curve')
    if len(arcana_mean_importances) > 0:
        longest_event_info = event_meta_data[event_meta_data['duration'] == event_meta_data['duration'].max()]
        longest_event_index = longest_event_info.index[0]
        viz.plot_arcana_mean_importances(importances=arcana_mean_importances[longest_event_index],
                                         top_n_features=min(5, train_data.shape[1]),
                                         ax=axs[1, 1])
        title = f'ARCANA-Importances {event_meta_data.at[longest_event_index, "start"]} - ' \
                f'{event_meta_data.at[longest_event_index, "end"]}'
        axs[1, 1].set_title(title)

        for i, arcana_mean_importance in enumerate(arcana_mean_importances):
            new_fig, ax = viz.plot_arcana_mean_importances(importances=arcana_mean_importance,
                                                           top_n_features=min(5, train_data.shape[1]))
            title = f'ARCANA-Importances {event_meta_data.at[i, "start"]} - {event_meta_data.at[i, "end"]}'
            ax.set_title(title)
            filename = (f'./arcana_importances_{i}.png' if save_dir is None
                        else os.path.join(save_dir, f'arcana_importances_{i}.png'))
            new_fig.savefig(filename, format='png')
            plt.close(fig=new_fig)

        if len(arcana_losses) > 0:
            # If Arcana losses are given, do loss plots for debugging
            viz.plot_arcana_losses(losses=arcana_losses[longest_event_index])
    else:
        axs[1, 1].text(0.5, 0.5, "No anomaly events detected.",
                       ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round,pad=0.5',
                                                                        facecolor='white',
                                                                        edgecolor='black',
                                                                        linewidth=1.5)
                       )
    plt.tight_layout()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'results.png'), dpi=300)

    plt.show()
