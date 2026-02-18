
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, Tuple, List
import warnings
import pickle
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# # pylint: disable=E0401,E0611,C0413
from tensorflow.keras.models import load_model as load_keras_model, Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, Callback

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin

DataType = Union[np.ndarray, pd.DataFrame]
logger = logging.getLogger('energy_fault_detector')


# TODO: reduce number of args of constructor (replace with kwargs, makes it easier to add others)
class Autoencoder(ABC, SaveLoadMixin):
    """Autoencoder template. Compatible with sklearn and keras/tensorflow.

    Args:
        learning_rate: learning rate of the adam optimizer. Default: 0.001.
        batch_size: number of samples per batch. Default: 128.
        epochs: number of epochs to run. Default: 10.
        loss_name: name of loss metric to use. Default: 'mean_squared_error'.
        metrics: list of additional metrics to track. Default: ['mean_absolute_error'].
        decay_rate: learning rate decay. Optional. If not defined, a fixed learning rate is used. Default: None.
        decay_steps: number of steps to decay learning rate over. Optional. Default: None.
            If not defined, a fixed learning rate is used.
        early_stopping: If True the early stopping callback will be used in the fit method. Early stopping will
            interrupt the training procedure before the last epoch is reached if the loss is not improving.
            The exact time of the interruption is based on the patience parameter.
            Default: False.
        patience: parameter for early stopping. If early stopping is used the training will end if more than
            patience epochs in a row have not shown an improved loss. Default: 5.
        min_delta: parameter of the early stopping callback. If the losses of an epoch and the next epoch differ
            by less than min_delta, they are considered equal (i.e. no improvement). Default: 1e-4.
        noise: float value that determines the influence of the noise term on the training input. High values mean
            highly noisy input. 0 means no noise at all. If noise >0 is used validation metrics will not be
            affected by it. Thus training loss and validation loss can differ depending on the magnitude of noise.
            Default: 0.0 (no noise).
        conditional_features: (optional) List of features to use as conditions for the conditional autoencoder.
            Default: None (no conditional features).
        verbose: verbosity mode - passed to KerasModel fit and predict methods. Default: 1.


    Attributes:
        is_conditional: Indicates whether the autoencoder is a conditional autoencoder or not.
        model: Keras Model created by the `self.create_model()` method.
        encoder: Keras Model created by the `self.create_model()` method.
        history: Dictionary with the loss, val_loss and metrics, if the model was fitted.
        callbacks: List of callbacks applied every epoch when fitting the model. If early stopping is enabled,
            contains at least the `keras.callbacks.early_stopping` callback.

    """

    def __init__(self,
                 *,
                 learning_rate: float = 0.001, batch_size: int = 128, epochs: int = 10,
                 loss_name: str = 'mean_squared_error', metrics: List[str] = None,
                 decay_rate: float = None, decay_steps: float = None,
                 early_stopping: bool = False, patience: int = 5, min_delta: float = 1e-4,
                 noise: float = 0.0,
                 conditional_features: Optional[List[str]] = None,
                 verbose: int = 1,
                 **kwargs):
        super().__init__()

        if kwargs:
            logger.warning("Unknown Autoencoder kwargs ignored: %s", list(kwargs.keys()))

        self.learning_rate = (
            ExponentialDecay(initial_learning_rate=learning_rate, decay_rate=decay_rate, decay_steps=decay_steps)
            if decay_rate and decay_steps else learning_rate
        )

        self.conditional_features: Optional[List[str]] = conditional_features
        self.is_conditional: bool = conditional_features is not None

        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.loss_name: str = loss_name  # TODO: allow custom loss and metrics functions?
        self.metrics: List[str] = ['mean_absolute_error'] if metrics is None else metrics
        self.noise: float = noise
        self.verbose: int = verbose

        self.model: Optional[KerasModel] = None
        self.encoder: Optional[KerasModel] = None
        self.decoder: Optional[KerasModel] = None
        self.history: Any = None

        self.callbacks: List[Callback] = [
            EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, restore_best_weights=True)
        ] if early_stopping else []

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        status = "built" if self.model is not None else "unbuilt"
        params = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "loss_name": self.loss_name,
            "metrics": self.metrics,
            "noise": self.noise,
            "conditional_features": self.conditional_features,
        }
        # Keep it compact
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{cls}({param_str}, model={status})"

    def summary(self, **kwargs) -> None:
        """Print a Keras summary of the underlying model, if built."""
        if self.model is None:
            print(f"{self.__class__.__name__}: model is not built yet.")
        else:
            self.model.summary(**kwargs)

    def __call__(self, x: Union[np.ndarray, tf.Tensor], conditions: Union[np.ndarray, tf.Tensor] = None) -> tf.Tensor:
        """Calls the model on new inputs."""
        if self.is_conditional:
            if conditions is None:
                raise ValueError('To call an conditional autoencoder on new input, the conditions need to be provided'
                                 ' as well: `Autoencoder(inputs, conditions)`.')
            return self.model([x, conditions])

        return self.model(x)

    @abstractmethod
    def create_model(self, input_dimension: Union[int, Tuple], condition_dimension: Optional[int] = None, **kwargs
                     ) -> KerasModel:
        """Create a keras model, sets the model and (optionally) encoder attributes.

        Args:
            input_dimension: number of features in input data.
            condition_dimension: number of features in conditional data. Only used if self.is_conditional = True.

        Returns:
            A Keras model.
        """

    def compile_model(self, new_learning_rate: float = None, **kwargs):
        """Compile (or recompile) model with Adam optimizer, optionally with a different learning rate."""

        if self.model is None:
            raise ValueError('You need to create the model first by calling the `create_model` method.')

        learning_rate = new_learning_rate if new_learning_rate else self.learning_rate
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.loss_name, metrics=self.metrics)

    def fit(self,
            x: DataType,
            x_val: DataType = None,
            **kwargs) -> 'Autoencoder':  # pylint: disable=W0237
        """Fit the autoencoder model.

        Args:
            x: training data
            x_val: validation data

        Returns:
            Fitted model.
        """

        if self.is_conditional and (self.conditional_features is None or len(self.conditional_features) == 0):
            raise ValueError('A list of conditional features must be provided for conditional autoencoders!')

        if self.model is None:
            self.create_model(
                input_dimension=x.shape[1] - len(self.conditional_features) if self.is_conditional else x.shape[1],
                condition_dimension=len(self.conditional_features) if self.is_conditional else None
            )

        self.compile_model()

        if 'callbacks' in kwargs:
            self.callbacks += kwargs['callbacks']
            kwargs.pop('callbacks')

        # ensure verbose default
        kwargs.setdefault("verbose", self.verbose)

        self._fit_internal(x, x_val, epochs=self.epochs, callbacks=self.callbacks, **kwargs)
        return self

    def _fit_internal(self, x: DataType, x_val: DataType, epochs: int, callbacks: List[Callback], **kwargs) -> None:
        """Fit the keras model on provided training data.

        Args:
            x: training data
            x_val: validation data
            epochs: number of epochs to run.
            callbacks: List of callbacks applied every epoch when fitting the model.
            kwargs: Additional arguments passed to the `fit` method.
        """

        if self.is_conditional:
            input_data, conditions, val_input_data, val_conditions = split_inputs(self.conditional_features, x, x_val)
            ae_input = (self._apply_noise(input_data), conditions)
            ae_target = input_data
            val_ae_input = (val_input_data, val_conditions)
            val_ae_target = val_input_data
        else:
            ae_input = self._apply_noise(x)
            ae_target = x
            val_ae_input = val_ae_target = x_val

        fit_history = self.model.fit(
            ae_input, ae_target,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=None if x_val is None else (val_ae_input, val_ae_target),
            callbacks=callbacks,
            **kwargs
        )

        self._extend_fit_history(fit_history.history)

    def tune(self, x: DataType, learning_rate: float, tune_epochs: int, x_val: DataType = None,
             **kwargs) -> 'Autoencoder':
        """Tune full autoencoder by extending the model fitting process by tune_epochs.

        Args:
            x: training data
            x_val: validation data
            learning_rate: learning rate to use during tuning.
            tune_epochs: number of epochs to tune.
            kwargs: other keyword args for the keras `Model.fit` method.

        Returns:
            Tuned model.
        """

        self.compile_model(learning_rate)  # sets new learning rate
        kwargs.setdefault("verbose", self.verbose)
        self._fit_internal(
            x, x_val,
            epochs=tune_epochs + self.epochs,
            callbacks=self.callbacks,
            initial_epoch=self.epochs,
            **kwargs
        )
        return self

    def tune_decoder(self, x: pd.DataFrame, learning_rate: float, tune_epochs: int, x_val: pd.DataFrame = None,
                     **kwargs) -> 'Autoencoder':
        """Tune decoder only - weights of the encoder are unchanged. Weight tuning is done by extending the model
        fitting process by tune_epochs.

        Args:
            x: Training data
            x_val: Validation data
            learning_rate: Learning rate to use during tuning. Default original learning rate.
            tune_epochs: Number of epochs to tune.
            kwargs: Other keyword args for the keras `Model.fit` method.

        Returns:
            Tuned model.
        """

        if self.encoder is None:
            raise ValueError("Encoder was not created. Update the `self.create_model` method to set the `self.encoder`"
                             " attribute.")

        self.encoder.trainable = False
        kwargs.setdefault("verbose", self.verbose)
        self.tune(x=x, x_val=x_val, learning_rate=learning_rate, tune_epochs=tune_epochs, **kwargs)
        return self

    def encode(self, x: DataType, conditions: DataType = None, **kwargs) -> np.ndarray:
        """Return latent representation using autoencoder."""

        if self.encoder is None:
            raise ValueError("Encoder was not created. Update the `self.create_model` method to set the `self.encoder`"
                             " attribute.")

        kwargs.setdefault("verbose", self.verbose)

        if self.is_conditional:
            if conditions is None:
                input_data, conditions, _, _ = split_inputs(self.conditional_features, x)
            return self.encoder.predict([input_data, conditions], **kwargs)

        return self.encoder.predict(x, **kwargs)

    def predict(self, x: DataType, **kwargs) -> DataType:
        """Predict values using fitted model.

        Args:
            x: input data
        """

        if not self._is_fitted():
            raise ValueError(f'{self.__class__} object needs to be fitted first!')

        kwargs.setdefault("verbose", self.verbose)
        return self._predict(x, **kwargs)

    def _predict(self, x: DataType, return_conditions: bool = False, **kwargs) -> DataType:
        """Predict with fitted model.

        Args:
            x: input data
            return_conditions: Whether to return conditional data or not. Defaults to False.
            kwargs: other keyword args for the keras `Model.predict` method.

        Returns:
            AE reconstruction of the input data.
        """
        if not self.is_conditional:
            if isinstance(x, pd.DataFrame):
                return pd.DataFrame(self.model.predict(x, **kwargs), index=x.index, columns=x.columns)
            return self.model.predict(x, **kwargs)

        input_data, conditions, _, _ = split_inputs(self.conditional_features, x)
        predictions = self.model.predict([input_data, conditions], **kwargs)
        prediction_df = pd.DataFrame(predictions, index=input_data.index, columns=input_data.columns)

        if return_conditions:
            input_data, conditions, _, _ = split_inputs(self.conditional_features, x)
            return conditions.join(prediction_df)

        return prediction_df

    def get_reconstruction_error(self, x: DataType, **kwargs) -> DataType:
        """Get the reconstruction error: output - input.

        Args:
            x: input data
            kwargs: other keyword args for the keras `Model.predict` method.

        Returns:
            AE reconstruction error of the input data.
        """

        x_predicted = self.predict(x, **kwargs)

        if self.is_conditional:
            prediction = x_predicted
            input_data, conditions, _, _ = split_inputs(self.conditional_features, x)
            recon_error = prediction - input_data
            if isinstance(x, pd.DataFrame):
                recon_error = pd.DataFrame(recon_error, index=input_data.index, columns=input_data.columns)
            return recon_error

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(x_predicted - x, index=x.index, columns=x.columns)

        return x_predicted - x

    def save(self, directory: str, overwrite: bool = False, **kwargs):  # pylint: disable=W0221,W0613
        """Save the model object in given directory, filename is the class name.

        Args:
            directory: directory to save the object in.
            overwrite: whether to overwrite existing data, default False.
        """

        self._create_empty_dir(directory, overwrite)
        file_path = os.path.join(directory, self.__class__.__name__)

        if self.model is not None:
            self.model.save(file_path + '.model')

        if self.encoder is not None:
            self.encoder.save(file_path + '.encoder')

        ae_dict = self.__dict__.copy()
        ae_dict['model'] = None
        ae_dict['encoder'] = None
        file_name = file_path + '.attrs'
        with open(file_name, 'wb') as f:
            f.write(pickle.dumps(ae_dict))

    def load(self, directory: str, **kwargs) -> 'Autoencoder':  # pylint: disable=W0221,W0613
        """Load the model object from given directory."""

        file_name = os.path.join(directory, self.__class__.__name__)
        with open(file_name + '.attrs', 'rb') as f:
            class_data = f.read()

        self.__dict__ = pickle.loads(class_data)
        if os.path.exists(file_name + '.model'):
            self.model = load_keras_model(file_name + '.model')
            if os.path.exists(file_name + '.encoder'):
                self.encoder = load_keras_model(file_name + '.encoder')
        else:
            warnings.warn('No fitted model was found.')

        return self

    def _is_fitted(self) -> bool:
        """Check whether fit was called at least once."""

        if self.model is None or self.history is None:
            return False
        return True

    def _extend_fit_history(self, tune_history: Dict[str, List[Any]]) -> None:
        """Extend the fit history."""
        if self.history is None:
            self.history = tune_history
            return

        for k, _ in self.history.items():
            if k in tune_history:
                self.history[k] = self.history[k] + tune_history[k]

    def _apply_noise(self, x: DataType) -> DataType:
        """Apply random normal noise - for denoising AEs.

        Args:
            x: Input data, which can be a NumPy array or a Pandas DataFrame.

        Returns:
            The input data with noise applied.
        """

        if self.noise == 0:
            return x

        return x + self.noise * np.random.normal(loc=0., scale=1.0, size=x.shape)

    def _apply_noise_generator(self, x):
        """Apply random normal noise to generator inputs - for denoising AEs.
        We assume the generator outputs inputs, outputs tuples."""

        if self.noise == 0:
            for batch in x:
                yield batch
            return

        for batch_input, batch_output in x:
            noisy_input = []
            for seq in batch_input:
                noisy_seq = seq + np.random.normal(0, self.noise, seq.shape)
                noisy_input.append(noisy_seq)

            yield np.array(noisy_input), batch_output


def split_inputs(conditional_features: List[str], x: pd.DataFrame, x_val: pd.DataFrame = None) -> Tuple:
    """Prepare the input and conditional data.

    Args:
        conditional_features (List[str]): List of features names that are used as condition.
        x (pd.DataFrame): Data to split.
        x_val (pd.DataFrame, optional): Validation data to split. Defaults to None.

    Returns:
        Tuple: Tuple of input and conditional data.
            input_data, conditions, val_input_data, val_conditions
    """

    def _split(data):
        if isinstance(data, pd.DataFrame):
            conditions = data[conditional_features]
            inputs = data.drop(conditional_features, axis=1)
        elif isinstance(data, (np.ndarray, tf.Tensor)):
            # Assume the first columns are the conditional features
            inputs = data[:, len(conditional_features):]
            conditions = data[:, :len(conditional_features)]
        else:
            raise ValueError("Input must be a pandas DataFrame, NumPy array, or TensorFlow Tensor.")
        return inputs, conditions

    input_data, conditions = _split(x)
    val_input_data, val_conditions = _split(x_val) if x_val is not None else (None, None)

    return input_data, conditions, val_input_data, val_conditions
