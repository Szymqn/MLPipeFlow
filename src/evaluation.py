from abc import ABC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.pipeline import PipelineComponent


class ModelEvaluation(PipelineComponent, ABC):
    def __init__(self):
        pass

    def execute(self, model, data):
        print('Evaluating model...')
        features, target = data
        predictions = model.predict(features)

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