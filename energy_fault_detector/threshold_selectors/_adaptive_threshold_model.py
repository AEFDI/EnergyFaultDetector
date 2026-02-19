import numpy as np
from keras import Input, Model as KerasModel
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from keras.src.optimizers.adam import Adam


class RegressionNN:
    """ Simple neural network (NN) with one hidden layer of shape (input_dim, size) used for regression.
    It uses ReLU-activation and the Adam Optimizer

    Args:
        Args:
        size (int): Hyperparameter determining the size of the hidden layer of the NN.
        learning_rate (float): Hyperparameter for the learning rate of the optimizer during training.
        batch_size (int): Number of samples per gradient update.
        early_stopping (bool): If True, the early stopping callback will be used in the fit method. Early stopping will
            interrupt the training procedure before the last epoch is reached if the loss is not improving.
            The exact time of the interruption is based on the patience parameter.
        patience (int): Parameter for early stopping. If early stopping is used, the training will end if more than
            `patience` epochs in a row have not shown an improved loss.
        validation_split (float): Fraction of the training data to be used as validation data (between 0 and 1).
    """
    def __init__(self, size: int, learning_rate: float, batch_size: int, early_stopping: bool, patience: int,
                 validation_split: float):
        self.size = size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.fit_history = None
        self.validation_split = validation_split
        if early_stopping and self.validation_split > 0:
            self.callbacks = [
                EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=patience, restore_best_weights=True)
            ]
        else:
            self.callbacks = None

    def create_model(self, input_dimension) -> None:
        """"Creates a keras model of a dense autoencoder with one hidden layer. """
        inputs = Input(shape=(input_dimension,))
        hidden_layer = Dense(self.size, activation="relu")(inputs)
        outputs = Dense(1, activation="linear")(hidden_layer)
        self.model = KerasModel(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

    def fit(self, x: np.array, y: np.array, epochs: int, verbose: int = 0) -> None:
        """ Fits the neural network model to the training data.

        Args:
            x (np.array): Training input data.
            y (np.array): Training targets (one-dimensional).
            epochs (int): Number of epochs for the NN training.
            verbose: determines the amount of console output of the NN training: 0=silent, 1=progress bar,
                2=one line per epoch
        """
        if self.model is None:
            self.create_model(x.shape[1])
        self.fit_history = self.model.fit(
            x, y, batch_size=self.batch_size, epochs=epochs,
            verbose=verbose, callbacks=self.callbacks, validation_split=self.validation_split
        )

    def predict(self, x: np.array, verbose: int = 0) -> np.ndarray:
        """ Predicts outcomes based on input data using the trained neural network model.

        Args:
            x (np.array): Input data for predictions.
            verbose: determines the amount of console output of the NN training: 0=silent, 1=progess bar,
                    2=one line per epoch

        Returns:
            np.ndarray: Predicted values from the neural network.
        """
        return self.model.predict(x=x, verbose=verbose)
