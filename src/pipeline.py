from abc import ABC, abstractmethod


class PipelineComponent(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs):
        pass
