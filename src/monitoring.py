from abc import ABC
from src.pipeline import PipelineComponent
from pickle import load


class Monitoring(PipelineComponent, ABC):
    def __init__(self):
        pass

    def execute(self, model_path, data):
        print('Monitoring model...')
        print("Model path: ", model_path)
        print("Data:", data)
        with open(model_path, 'rb') as f:
            model = load(f)
        print("Model: ", model)
        features, target = data
        if hasattr(model, 'predict'):
            predictions = model.predict(features)
            return predictions
        else:
            raise AttributeError(f"Loaded object from {self.model_path} does not have a 'predict' method.")
