from abc import ABC
from pipeline import PipelineComponent
from sklearn.linear_model import LogisticRegression


class ModelTraining(PipelineComponent, ABC):
    def __init__(self):
        self.model = LogisticRegression()

    def execute(self, data):
        features, target = data
        self.model.fit(features, target)
        return self.model
