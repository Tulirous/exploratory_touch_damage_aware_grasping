from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from human_mimic_demo.messages import TrackingSample


class HandTracker(ABC):
    @abstractmethod
    def read(self) -> Optional[TrackingSample]:
        """Return the newest right-hand observation or None when tracking is lost."""

    def close(self) -> None:
        pass

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
