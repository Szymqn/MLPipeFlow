# MLPipeFlow

## Description
MLPipeFlow is a machine learning pipeline framework that simplifies the process of data preprocessing, feature engineering, model training, evaluation, deployment, and monitoring. It is designed to be modular and extensible, allowing users to easily integrate their own components.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the pipeline, create an instance of `MLPipeFlow` with the desired components and call the `run` method:
```python
from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.evaluation import ModelEvaluation
from src.training import ModelTraining
from src.deployment import ModelDeployment
from src.monitoring import Monitoring
from src.mlpipeline import MLPipeFlow

components = [
    DataPreprocessing(data_path='path/to/data.csv'),
    FeatureEngineering(target='target_column'),
    ModelTraining(),
    ModelEvaluation(),
    ModelDeployment(),
    Monitoring()
]

pipeline = MLPipeFlow(components)
results = pipeline.run()
print(results)
```

## Components
### DataPreprocessing
Handles loading and preprocessing of data.
- **Methods**:
  - `load_data()`: Loads data from the specified path.
  - `execute()`: Executes the preprocessing steps.

### FeatureEngineering
Handles feature engineering tasks.
- **Methods**:
  - `execute(data)`: Performs feature engineering on the provided data.

### ModelTraining
Handles model training.
- **Methods**:
  - `execute(data)`: Trains the model using the provided data.

### ModelEvaluation
Handles model evaluation.
- **Methods**:
  - `execute(model, data)`: Evaluates the model using the provided data.

### ModelDeployment
Handles model deployment.
- **Methods**:
  - `execute(model)`: Deploys the trained model.

### Monitoring
Handles monitoring of the deployed model.
- **Methods**:
  - `execute(model_path, data)`: Monitors the deployed model.

## Technologies Used
- Python
- scikit-learn
- pandas
- NumPy

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License
This project is licensed under the MIT License.
