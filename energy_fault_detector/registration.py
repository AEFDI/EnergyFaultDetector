"""Class for registering classes of different types."""

import importlib
from typing import TypeVar, List, Dict

SomeClass = TypeVar('SomeClass')


class Registry:
    """Registry for classes to get the class by name and type."""

    def __init__(self):
        self.registry: Dict[str, Dict[str, str]] = {}
        self.class_to_class_names: Dict[str, List[str]] = {}

    def register(self, module_path: str, class_type: str, class_names: List[str]) -> None:
        """Register a class to refer to in the configuration.

        Args:
            module_path (str): Path to the module containing the class.
            class_type (str): type of the class (anomaly_score, autoencoder, threshold_selector or data_preprocessor)
            class_names (List[str], optional): a list of names which may be used to refer to this class. Default is None
        """

        self._check_exists(class_type, class_names)
        self.registry.setdefault(class_type, {})
        for class_name in class_names:
            self.registry[class_type][class_name] = module_path

        # if class name not in the class names, add this as well
        actual_class_name = module_path.split('.')[-1]
        self.class_to_class_names[actual_class_name] = class_names

    def get(self, class_type: str, class_name: str) -> SomeClass:
        """Get a class by class_type and class_name from the registry"""
        try:
            module_path = self.registry[class_type][class_name]
        except KeyError as e:
            raise ValueError(f'The class {class_name} of type {class_type} is not known.') from e
        module_name, actual_class_name = module_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, actual_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f'Could not load {actual_class_name} from {module_name}.') from e

    def get_alternative_class_names(self, class_name: str) -> List[str]:
        """Get all alternative class names of the given class"""
        try:
            return self.class_to_class_names[class_name]
        except KeyError as e:
            raise ValueError(f'The class {class_name} is not known.') from e

    def _check_exists(self, class_type: str, class_names: List[str]):
        """Check whether there is already a class registered under this type and name."""
        if self.registry.get(class_type):
            for class_name in class_names:
                if self.registry[class_type].get(class_name):
                    raise ValueError(f'Class {self.registry[class_type][class_name]} is already registered under '
                                     f'class type {class_type} and class name {class_name}.')

    def print_available_classes(self) -> None:
        """Print the available classes and their config names"""
        for class_type in self.registry:
            temp = {}
            for key, value in self.registry[class_type].items():
                if value in temp:
                    temp[value].append(key)
                else:
                    temp[value] = [key]
            print(f'\nAvailable classes for {class_type}:')
            for module_path, names in temp.items():
                print(f'Class: {module_path} \nConfig names: {names}')


# global registry
registry: Registry = Registry()


def register(**kwargs):
    """Shorthand function for registration"""
    registry.register(**kwargs)
