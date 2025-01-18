from abc import ABC
from pipeline import PipelineComponent
from joblib import load


class Monitoring(PipelineComponent, ABC):
    def __init__(self, model_path):
        self.model_path = model_path

    def execute(self, data):
        model = load(self.model_path)
        predictions = model.predict(data)
        return predictions
