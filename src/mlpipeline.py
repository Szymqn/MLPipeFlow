from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.evaluation import ModelEvaluation
from src.monitoring import Monitoring
from src.training import ModelTraining
from src.deployment import ModelDeployment


class MLPipeFlow:
    def __init__(self, components):
        self._components = components

    def run(self):
        global model_path, evaluation_results, monitoring_results
        data = None
        model = None
        for component in self._components:
            if isinstance(component, DataPreprocessing):
                data = component.execute()
            elif isinstance(component, FeatureEngineering):
                data = component.execute(data)
            elif isinstance(component, ModelTraining):
                model = component.execute(data)
            elif isinstance(component, ModelEvaluation):
                evaluation_results = component.execute(model, data)
            elif isinstance(component, ModelDeployment):
                model_path = component.execute(model)
            elif isinstance(component, Monitoring):
                monitoring_results = component.execute(model_path, data)
        return {
            'evaluation_results': evaluation_results,
            'monitoring_results': monitoring_results
        }
