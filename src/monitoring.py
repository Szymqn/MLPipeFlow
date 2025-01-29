from abc import ABC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.pipeline import PipelineComponent
from pickle import load


class Monitoring(PipelineComponent, ABC):
    def __init__(self):
        pass

    def execute(self, model_path, data):
        print('Monitoring model...')
        with open(model_path, 'rb') as f:
            model = load(f)
        features, target = data
        if hasattr(model, 'predict'):
            predictions = model.predict(features)
        else:
            raise AttributeError(f"Loaded object from {self.model_path} does not have a 'predict' method.")

        acc_score = accuracy_score(target, predictions)
        precision = precision_score(target, predictions, average='binary')
        recall = recall_score(target, predictions, average='binary')
        f1 = f1_score(target, predictions, average='binary')

        print("ACC: ", acc_score)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1)

        return {
            'accuracy': acc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
