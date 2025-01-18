from abc import ABC
from pipeline import PipelineComponent
from joblib import dump


class ModelDeployment(PipelineComponent, ABC):
    def __init__(self, model, model_path):
        self.model = model
        self.model_path = model_path

    def execute(self, data):
        dump(self.model, self.model_path)
        return self.model_path
