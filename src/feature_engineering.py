from abc import ABC
from src.pipeline import PipelineComponent


class FeatureEngineering(PipelineComponent, ABC):
    def __init__(self, target):
        self._target = target

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    def execute(self, data):
        print('Engineering features...')
        features = data.drop(columns=[self._target])
        target = data[self._target]
        return features, target
