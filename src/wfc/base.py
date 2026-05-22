from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseSolver(ABC):
    def __init__(self, library):
        self.lib = library

    @abstractmethod
    def solve(self, H:int, W:int, seed:Optional[int]=None, start_token:Optional[int]=None):
        pass
