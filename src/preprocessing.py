from abc import ABC
import pandas as pd
from src.pipeline import PipelineComponent


class DataPreprocessing(PipelineComponent, ABC):
    def __init__(self, data_path):
        self._data_path = data_path
        self._data = None

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        self._data_path = value

    def load_data(self):
        self._data = pd.read_csv(self._data_path)
        return self._data

    def execute(self):
        print('Preprocessing data...')
        return self.load_data()
