
import os
import shutil
import pickle


class SaveLoadMixin:
    """Save and load methods for model-like (regressors/classifiers/transformers) objects.

    Override the save and load functions when one of the attributes cannot be pickled, such as keras/tensorflow models.
    """

    def __init__(self, **kwargs):
        # **kwargs to ensure this class works fine in multiple inheritance cases.
        pass

    def save(self, directory: str, overwrite: bool = False, file_name: str = None):
        """Save the model object in given directory, filename is the class name.
        Note: override when using Keras/Tensorflow, as those models cannot be pickled.

        Args:
            directory: directory to save the object in.
            overwrite: whether to overwrite existing data, default False.
            file_name: name of the file to save the object in, if none, take the class name.
        """

        file_name = self.__class__.__name__ + '.pkl' if file_name is None else file_name
        path = os.path.join(directory, file_name)

        self._create_empty_dir(directory, overwrite, file_name)

        with open(path, 'wb') as f:
            f.write(pickle.dumps(self.__dict__))

    def load(self, directory: str, file_name: str = None):
        """Load the model object from given directory.
        Note: override when using Keras/Tensorflow, as those models cannot be pickled.
        """

        file_name = self.__class__.__name__ + '.pkl' if file_name is None else file_name
        path = os.path.join(directory, file_name)

        with open(path, 'rb') as f:
            class_data = f.read()

        self.__dict__ = pickle.loads(class_data)

    @staticmethod
    def _create_empty_dir(directory: str, overwrite: bool, file_name: str = None):
        """Create a new directory or, when overwrite=True, replace an existing one with a new empty directory if it
        already exists.

        Args:
            directory: directory to create.
            overwrite: whether to overwrite existing data, default False.
            file_name: (optional) Name of the file to save the object in. Defaults to None.
        """

        file_path = directory if file_name is None else os.path.join(directory, file_name)

        if os.path.exists(file_path) and overwrite:
            shutil.rmtree(directory)
        elif os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f'File {file_path} already exists!')

        os.makedirs(directory, exist_ok=True)
