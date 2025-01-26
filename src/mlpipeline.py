from src.evaluation import ModelEvaluation
from src.monitoring import Monitoring
from src.training import ModelTraining
from src.deployment import ModelDeployment


class MLPipeFlow:
    def __init__(self, components):
        self.components = components
        self.model_path = None

    def run(self):
        data = None
        model = None
        for component in self.components:
            if isinstance(component, ModelTraining):
                model = component.execute(data)
            elif isinstance(component, ModelEvaluation):
                data = component.execute(data, model)
            elif isinstance(component, ModelDeployment):
                self.model_path = component.execute(model)
            elif isinstance(component, Monitoring):
                component.execute(self.model_path, data)
            else:
                data = component.execute(data)
        return data
