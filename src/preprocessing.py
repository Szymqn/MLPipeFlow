import pandas as pd

from abc import ABC
from src.pipeline import PipelineComponent


class DataPreprocessing(PipelineComponent, ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    def execute(self, data=None):
        print('Data preprocessing...')
        return pd.read_csv(self.data_path)
