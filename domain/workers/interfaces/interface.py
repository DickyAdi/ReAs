from abc import ABC, abstractmethod


class WorkerInterface(ABC):
    @abstractmethod
    def run(self, **kwargs): ...
