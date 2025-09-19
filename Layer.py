from abc import ABC, abstractmethod
import numpy as np

# Layer interface
class Layer(ABC):
    @abstractmethod
    def process(self, input: np.ndarray) -> np.ndarray:
        """Process input and return output feature map"""
        pass
