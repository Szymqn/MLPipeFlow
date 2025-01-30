import os

from abc import ABC
from src.pipeline import PipelineComponent
from pickle import dump


class ModelDeployment(PipelineComponent, ABC):
    def __init__(self, model_path):
        self._model_path = model_path

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        self._model_path = value

    def execute(self, model):
        print('Deploying model...')
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        with open(self._model_path, 'wb') as f:
            dump(model, f)
        return self._model_path
