from abc import ABC
from src.pipeline import PipelineComponent
from sklearn.metrics import accuracy_score


class ModelEvaluation(PipelineComponent, ABC):
    def __init__(self):
        pass

    def execute(self, data, model):
        print('Evaluating model...')
        features, target = data
        predictions = model.predict(features)
        return accuracy_score(target, predictions)
