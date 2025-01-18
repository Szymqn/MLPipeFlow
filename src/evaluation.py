from abc import ABC
from pipeline import PipelineComponent
from sklearn.metrics import accuracy_score


class ModelEvaluation(PipelineComponent, ABC):
    def __init__(self, model):
        self.model = model

    def execute(self, data):
        features, target = data
        predictions = self.model.predict(features)
        return accuracy_score(target, predictions)
