from abc import ABC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from src.pipeline import PipelineComponent


class ModelTraining(PipelineComponent, ABC):
    def __init__(self, model_name):
        self._model = None
        self._model_name = model_name

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    def choose_model(self):
        match self._model_name:
            case 'logistic_regression':
                self._model = LogisticRegression()
            case 'linear_regression':
                self._model = LinearRegression()
            case 'random_forest':
                self._model = RandomForestClassifier()
            case 'svm':
                self._model = SVC()
            case 'knn':
                self._model = KNeighborsClassifier()
            case _:
                raise ValueError(f"Model {self._model_name} not supported")

    def execute(self, data):
        print('Training model...')
        self.choose_model()

        features, target = data
        self._model.fit(features, target)
        return self._model
