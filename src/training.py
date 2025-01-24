from abc import ABC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from src.pipeline import PipelineComponent


class ModelTraining(PipelineComponent, ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def choose_model(self, model_name):
        match model_name:
            case 'logistic_regression':
                self.model = LogisticRegression()
            case 'linear_regression':
                self.model = LinearRegression()
            case 'random_forest':
                self.model = RandomForestClassifier()
            case 'svm':
                self.model = SVC()
            case 'knn':
                self.model = KNeighborsClassifier()
            case _:
                raise ValueError(f"Model {model_name} not supported")

    def execute(self, data):
        print('Training model...')
        self.choose_model(self.model_name)

        features, target = data
        self.model.fit(features, target)
        return self.model
