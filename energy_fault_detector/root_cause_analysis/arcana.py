"""ARCANA implementation, as described in https://doi.org/10.1016/j.egyai.2021.100065 ."""


import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tensorflow.keras.optimizers import Adam

from energy_fault_detector.core import Autoencoder

logger = logging.getLogger('energy_fault_detector')

AE_TYPE = Autoencoder
BIAS_RETURN_TYPE = Tuple[tf.Variable, Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Variable]


class Arcana:
    """Anomaly root cause analysis. Tries to find which of the sensors/inputs caused
    the reconstruction error of an autoencoder model.

    This is done by minimizing the loss function:

        '(1 - alpha) L2(X_corr - autoencoder(X_corr)) + alpha * L1(X_corr - X_obs)'

    where alpha is a hyperparameter between 0 and 1, X_corr is the corrected observation
    signal (X_obs + X_bias, should be without anomaly after optimization) and X_obs the
    original signal. We are interested in finding the Arcana-correction X_bias, which
    indicates the deviation of each sensor causing the reconstruction error (the deviation
    from the case without anomaly, still close to the original observation X_obs).

    Minimizing the L2 term results in minimal deviations (small x_bias values), the L1
    term keeps the solution close to the original sensor values, effectively
    keeping the number of inputs responsible for the reconstruction error small.

    For optimization itself the Adam Optimizer from `tensorflow.keras.optimizers` is used.

    Args:
        model: Autoencoder model to consider. Must have a __call__ method expecting input data and returns a tf.Tensor
        learning_rate: Learning rate for the adam optimizer.
        init_x_bias: Where to start, one of 'recon' (reconstruction error), 'zero' (a zero vector), 'weightedA'
            (alpha * reconstruction error) or 'weightedB' ( (1-alpha) * reconstruction error).
            Default 'recon'.
        alpha: hyperparameter of arcana loss function.
            A high alpha value means the L1-loss is weighted more, which results in an x_bias with smaller values.
        num_iter: Number of times to run the AdamOptimizer.
        epsilon: Small number to prevent division by zero for the adam optimizer, default 1e-8
        verbose: Whether to log loss values every 50 iterations, default False
        max_sample_threshold: Maximum number of samples which are analyzed by ARCANA. This parameter ensures that
            ARCANA calculations are sufficiently fast. Default is 1000
        kwargs: Any other arguments of the optimizer

    Attributes:
        opt: Adam optimizer object

    Configuration example:

    .. code-block:: text

        root_cause_analysis:
          alpha: 0.8
          init_x_bias: recon
          num_iter: 200
          max_sample_threshold: 1000
          verbose: false
    """

    def __init__(self, model: AE_TYPE, learning_rate: float = 0.001, init_x_bias: str = 'recon',
                 alpha: float = 0.8, num_iter: int = 400, epsilon: float = 1e-8, verbose: bool = False,
                 max_sample_threshold: int = 1000, **kwargs):

        self.keras_model: AE_TYPE = model

        self.learning_rate: float = learning_rate
        self.epsilon: float = epsilon

        x_bias_init_settings: List[str] = ['recon', 'zero', 'weightedA', 'weightedB']
        if init_x_bias not in x_bias_init_settings:
            raise ValueError(f'unknown init_x_bias setting, must be one of {x_bias_init_settings}')

        self.opt: Adam = Adam(learning_rate=self.learning_rate, epsilon=epsilon, **kwargs)
        self.init_x_bias: str = init_x_bias
        self.alpha: float = alpha
        self.num_iter: int = num_iter
        self.verbose: bool = verbose
        self.max_sample_threshold = max_sample_threshold

    def find_arcana_bias(self, x: pd.DataFrame, track_losses: bool = False, track_bias: bool = False
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
        """Find correction to input data x necessary to minimize the Arcana loss
        function. Large (absolute) correction in one of the inputs means that
        this is an important sensor/input for the reconstruction error.
        Detects which input are probably anomalous and resulted in an anomaly.

        Args:
            x: pandas DataFrame containing data with timestamp as index.
            track_losses: If True losses will be returned as a dictionary containing lists of combined loss, loss 1 and
                loss 2 for each 50th iteration)
            track_bias: If True bias will be returned as a list arcana biases each 50th iteration)

        Returns:
            x_bias: pandas DataFrame
            tracked_losses: A dataframe containing the combined loss, loss 1 (reconstruction) and
                loss 2 (regularization) for each 50th iteration (if track_losses is False this list is empty)
            tracked_bias: A List of dataframes representing x_bias
        """

        conditions = None
        if self.keras_model.is_conditional:
            # split inputs for conditional models
            conditions = x[self.keras_model.conditional_features]
            x = x.drop(self.keras_model.conditional_features, axis=1)

        feature_names = x.columns
        timestamps = x.index
        x = x.values.astype('float32')

        selection = self.draw_samples(x, conditions)
        x = x[selection]
        timestamps = timestamps[selection]

        if self.keras_model.is_conditional:
            conditions = conditions.values[selection]
            conditions = tf.constant(conditions, dtype=tf.float32)

        x_bias = self.initialize_x_bias(x, conditions)
        x_bias = tf.Variable(x_bias, dtype=tf.float32)

        tracked_losses = {'Combined Loss': [], 'Reconstruction Loss': [], 'Regularization Loss': [], 'Iteration': []}
        tracked_bias = [x_bias.numpy()] if track_bias else []

        for i in range(self.num_iter):
            x_bias, losses, _ = self.update_x_bias(x, x_bias, conditions)

            if i % 50 == 0:
                loss_1, loss_2, combined_loss = losses[0].numpy(), losses[1].numpy(), losses[2].numpy()
                if track_losses:
                    tracked_losses['Iteration'].append(i)
                    tracked_losses['Combined Loss'].append(combined_loss)
                    tracked_losses['Reconstruction Loss'].append(loss_1)
                    tracked_losses['Regularization Loss'].append(loss_2)
                if track_bias:
                    bias = x_bias.numpy()
                    tracked_bias.append(bias)
                if self.verbose:
                    logger.info('%d Combined Loss: %.2f', i, combined_loss)

        x_bias = pd.DataFrame(data=x_bias.numpy(), columns=feature_names, index=timestamps)

        # return x_bias as a pandas DataFrame
        tracked_losses = pd.DataFrame(tracked_losses)
        tracked_losses = tracked_losses.set_index('Iteration')
        tracked_bias_dfs = [pd.DataFrame(data=bias, columns=feature_names, index=timestamps) for bias in tracked_bias]
        return x_bias, tracked_losses, tracked_bias_dfs

    def draw_samples(self, x: np.ndarray, conditions: np.ndarray = None) -> np.ndarray:
        """ Selects index values from 0 to data_length by choosing the indexes with the highest anomaly score,
        for defining the ARCANA samples.

        Args:
            x (np.ndarray): Data of which samples should be drawn
            conditions (np.ndarray): (optional) Array of conditional features values

        Returns:
            array of booleans defining the selected samples.
        """
        if len(x) > self.max_sample_threshold:
            recon_error = np.abs(self.keras_model(x, conditions) - x)
            anomaly_score = np.sqrt(np.mean(recon_error ** 2, axis=1))
            threshold_index = np.argsort(anomaly_score)[-self.max_sample_threshold]
            selection = anomaly_score >= anomaly_score[threshold_index]
        else:
            selection = np.full(fill_value=True, shape=(len(x),))
        return selection

    def initialize_x_bias(self, x: np.ndarray, conditions: np.ndarray = None) -> tf.Tensor:
        """Initialize the ARCANA bias vector.

        Args:
            x: numpy array containing input data.
            conditions: numpy array containing conditional data - for conditional autoencoders.
                Defaults to None.

        Returns:
            initial x_bias values
        """
        x_bias = None
        if self.init_x_bias == 'recon':
            x_bias = self.keras_model(x, conditions) - x
        elif self.init_x_bias == 'zero':
            x_bias = 0 * x
        elif self.init_x_bias == 'weightedA':
            x_bias = self.alpha * (0 * x) + (1 - self.alpha) * (self.keras_model(x, conditions) - x)
        elif self.init_x_bias == 'weightedB':
            x_bias = (1 - self.alpha) * (0 * x) + self.alpha * (self.keras_model(x, conditions) - x)

        return x_bias

    def update_x_bias(self, x: tf.Variable, x_bias: tf.Variable, conditions: tf.Tensor = None) -> BIAS_RETURN_TYPE:
        """This function builds a tensor which can calculate the ARCANA loss (full_loss) and computes the gradient
        of that loss with respect to the Variable x + x_bias using tensorflow GradientTape.

        Args:
            x: tensorflow variable containing input data
            x_bias: tensorflow variable containing the current ARCANA bias.
            conditions: (optional) tensor containing the conditional feature values.

        Returns:
            x_corrected (x_bias-x): tf.Variable
            losses: Tuple(tf.Variable, tf.Variable, tf.Variable) contains the losses of this x_bias update.
            grad: (tf.variable) contains the computed gradient for this x_bias update.
        """
        with tf.GradientTape() as grad_tape:
            # no need to watch x_corrected (we do everything with x_bias)
            x_sim = self.keras_model(x + x_bias, conditions)  # predict the corrected value of x

            # loss part 1: measures the degree of anomaly of the ARCANA-corrected x_corrected
            loss_1 = 0.5 * tf.reduce_mean((x_sim - x - x_bias) ** 2)
            # loss part 2: measures the norm of x_bias and therefore the deviation
            # how far the Arcana-correction is away from the observation x
            loss_2 = tf.reduce_mean(tf.abs(x_bias))
            # full loss: combination of loss 1 and loss 2 weighted by alpha
            loss_full = (1 - self.alpha) * loss_1 + self.alpha * loss_2

        # differentiate w.r.t. x_bias
        grad = grad_tape.gradient(loss_full, x_bias)
        self.opt.apply_gradients(zip([grad], [x_bias]))
        return x_bias, (loss_1, loss_2, loss_full), grad
