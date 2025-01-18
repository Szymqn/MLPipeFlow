import pandas as pd

from abc import ABC
from pipeline import PipelineComponent


class DataPreprocessing(PipelineComponent, ABC):
    def __init__(self, data_path):
        self.data = data_path

    def execute(self, data=None):
        return pd.read_csv(self.data)
