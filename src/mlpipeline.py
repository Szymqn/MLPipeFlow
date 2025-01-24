from src.evaluation import ModelEvaluation
from src.monitoring import Monitoring
from src.training import ModelTraining
from src.deployment import ModelDeployment


class MLPipeFlow:
    def __init__(self, components):
        self.components = components

    def run(self):
        data = None
        model = None
        for component in self.components:
            if isinstance(component, ModelTraining):
                model = component.execute(data)
            elif isinstance(component, ModelEvaluation):
                data = component.execute(data, model)
            elif isinstance(component, ModelDeployment):
                data = component.execute(model)
            elif isinstance(component, Monitoring):
                data = component.execute(data, model)
            else:
                data = component.execute(data)
        return data
