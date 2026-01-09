
from typing import Union

import optuna as op
import pandas as pd
import numpy as np

from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.config import Config


def automatic_hyper_opt(config: Config, train_data: pd.DataFrame, normal_index: Union[pd.Series, None],
                        pca_code_size: int, num_trials: int) -> dict:
    """ Uses optuna to optimize the autoencoder hyperparameters 'batch_size', 'epochs', 'learning_rate', 'layers', and
    'code_size' with respect to the MSE of the reconstructions on the validation data.

    Args:
        config (Config): Config object
        train_data (pd.DataFrame): DataFrame containing training data
        normal_index (Union[pd.Series, None]): Series containing boolean information about the training data status.
            Can also be omitted.
        pca_code_size (int): recommended code_size based on a 99% variance explaining PCA.
        num_trials (int): number of trials for the hyperparameter optimization

    Returns:
        dict: Dictionary containing parameter names as keys and optimized values as values.
    """
    batch_sizes = [32, 64, 128]
    epochs = [10, 20, 30, 40, 50]
    learning_rate = (1e-5,  # low
                     0.01  # high
                     )
    prepped_train_data, _, _ = FaultDetector(config).preprocess_train_data(sensor_data=train_data,
                                                                           normal_index=normal_index)
    input_dim = prepped_train_data.shape[1]
    # Define layer parameter ranges based on input dimension
    if input_dim <= 10:
        layers_0 = (10,  # low
                    40  # high
                    )
        layers_1 = (10,  # low
                    20  # high
                    )
        layers_2 = (5,  # low
                    10  # high
                    )
        code_size = (max(pca_code_size - 5, 1),  # low
                     pca_code_size  # high
                     )
    elif input_dim <= 20:
        layers_0 = (20,  # low
                    75  # high
                    )
        layers_1 = (25,  # low
                    50  # high
                    )
        layers_2 = (15,  # low
                    25  # high
                    )
        code_size = (max(pca_code_size - 5, 1),  # low
                     pca_code_size  # high
                     )
    else:
        layers_0 = (input_dim - int(input_dim / 5),  # low
                    input_dim + int(input_dim / 5)  # high
                    )
        num_neurons = int(input_dim / 2)
        layers_1 = (num_neurons - int(num_neurons / 5),  # low
                    num_neurons + int(num_neurons / 5)  # high
                    )
        num_neurons = int(input_dim / 4)
        layers_2 = (num_neurons - int(num_neurons / 5),  # low
                    num_neurons + int(num_neurons / 5)  # high
                    )
        code_size = (max(pca_code_size - int(pca_code_size / 5), 1),  # low
                     pca_code_size  # high
                     )

    def reconstruction_mse(trial: op.Trial) -> float:
        """Samples new hyperparameters. fits a new model and returns the reconstruction error (MSE) of the validation
        data.

        Args:
            trial: optuna Trial object

        Returns:
            MSE of the reconstruction on validation data.
        """

        params = config.config_dict['train']['autoencoder']['params']

        # sample new parameters
        params['batch_size'] = int(trial.suggest_categorical(
            name='batch_size', choices=batch_sizes
        ))
        params['learning_rate'] = trial.suggest_float(
            name='learning_rate', low=learning_rate[0], high=learning_rate[1], log=True
        )
        params['epochs'] = int(trial.suggest_categorical(
            name='epochs', choices=epochs
        ))

        # architecture
        params['layers'][0] = trial.suggest_int(
            name='layers_0', low=layers_0[0], high=layers_0[1]
        )
        params['layers'][1] = trial.suggest_int(
            name='layers_1', low=layers_1[0], high=layers_1[1]
        )
        params['layers'][1] = trial.suggest_int(
            name='layers_2', low=layers_2[0], high=layers_2[1]
        )
        params['code_size'] = trial.suggest_int(
            name='code_size', low=code_size[0], high=code_size[1]
        )

        # update the configuration
        config.update_config(config.config_dict)

        # create a new model using our new configuration and train the model
        model = FaultDetector(config)
        # For autoencoder optimization, we do not need to fit a threshold
        training_dict = model.fit(train_data, normal_index=normal_index, fit_autoencoder_only=True,
                                  save_model=False)

        # Calculate the MSE of the reconstruction errors of the validation data - this is minimized
        deviations = training_dict.val_recon_error
        score = float(np.mean((np.square(deviations))))

        # help garbage collection
        del model

        return score

    study = op.create_study(sampler=op.samplers.TPESampler(),
                            study_name='autoencoder_optimization',
                            direction='minimize')

    autoencoder_params = config.config_dict['train']['autoencoder']['params']
    study.enqueue_trial(params={
        'batch_size': autoencoder_params['batch_size'],
        'learning_rate': autoencoder_params['learning_rate'],
        'layers_0': input_dim,
        'layers_1': int(input_dim / 2),
        'layers_2': int(input_dim / 4),
        'code_size': pca_code_size,
    })

    study.optimize(reconstruction_mse, n_trials=num_trials)

    best_params = study.best_params
    layers = [best_params.pop('layers_0'), best_params.pop('layers_1'), best_params.pop('layers_2')]
    best_params['layers'] = layers
    return best_params
