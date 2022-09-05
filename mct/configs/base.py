from abc import ABC, abstractmethod

class ConfigLoader(ABC):
    def __init__(self, config_path):
        self._config_path = config_path
    @abstractmethod
    def loader(self):
        pass