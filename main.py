from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.training import ModelTraining
from src.evaluation import ModelEvaluation
from src.deployment import ModelDeployment
from src.monitoring import Monitoring
from src.mlpipeline import MLPipeFlow


def main():
    components = [
        DataPreprocessing('data.csv'),
        FeatureEngineering(),
        ModelTraining(),
        ModelEvaluation('model.pkl'),
        ModelDeployment('model.pkl', 'example-model/model.pkl'),
        Monitoring('model.pkl')
    ]

    pipeline = MLPipeFlow(components)
    result = pipeline.run()
    print(f'Final Result: {result}')


if __name__ == "__main__":
    main()
