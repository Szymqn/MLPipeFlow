import os

from abc import ABC
from src.pipeline import PipelineComponent
from pickle import dump


class ModelDeployment(PipelineComponent, ABC):
    def __init__(self, model_path):
        self.model_path = model_path

    def execute(self, model):
        print('Deploying model...')
        print('Model', model)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            dump(model, f)

        return self.model_path
