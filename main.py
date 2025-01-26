from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.training import ModelTraining
from src.deployment import ModelDeployment
from src.evaluation import ModelEvaluation
from src.mlpipeline import MLPipeFlow


def main():
    components = [
        DataPreprocessing(data_path='test_data/exampleData_TCGA_LUAD_2000.csv'),
        FeatureEngineering(target='class'),
        ModelTraining(model_name='random_forest'),
        ModelEvaluation(),
        ModelDeployment(model_path='model_example/model.pkl'),
    ]

    pipeline = MLPipeFlow(components)
    result = pipeline.run()
    print(f'Final Result: {result}')


if __name__ == "__main__":
    main()
