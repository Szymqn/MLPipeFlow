from abc import ABC
from pipeline import PipelineComponent
from sklearn.preprocessing import StandardScaler


class FeatureEngineering(PipelineComponent, ABC):
    def __init__(self):
        self.scaler = StandardScaler()

    def execute(self, data):
        features = data.drop(columns=['target'], axis=1)
        target = data['target']
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, target
