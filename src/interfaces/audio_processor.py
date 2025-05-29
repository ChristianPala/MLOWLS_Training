from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AudioProcessor(ABC):
    """Abstract interface for audio preprocessing."""

    @abstractmethod
    def process_file(self, audio_path: str, **kwargs: Any) -> tuple[list[np.ndarray], list[float]]:
        """Process audio file into segments."""
        pass

    @abstractmethod
    def process_bytes(
        self, audio_bytes: bytes, **kwargs: Any
    ) -> tuple[list[np.ndarray], list[float]]:
        """Process audio from bytes."""
        pass
