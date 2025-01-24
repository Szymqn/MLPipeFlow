from abc import ABC
from src.pipeline import PipelineComponent
from sklearn.preprocessing import StandardScaler


class FeatureEngineering(PipelineComponent, ABC):
    def __init__(self, target):
        self.scaler = StandardScaler()
        self.target = target

    def execute(self, data):
        print('Feature engineering...')
        if self.target is None:
            raise ValueError('Target column name is required')

        features = data.drop(columns=[self.target], axis=1)
        target = data[self.target]
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, target
