import pytest
from unittest.mock import patch, MagicMock
from mlflow_data.experiment import run_experiment

@patch("mlflow_data.experiment.mlflow.set_experiment")
@patch("mlflow_data.experiment.subprocess.run")
def test_run_experiment_knn(mock_subprocess_run, mock_set_experiment):
    # Mock subprocess.run and mlflow.set_experiment
    mock_subprocess_run.return_value = MagicMock()
    mock_set_experiment.return_value = None

    # Call the function with KNN model type
    run_experiment(
        data_path="data/dataset_Cramer.parquet",
        model_type="KNN",
        n_neighbors=5,
        weights="distance",
        retrain=False
    )

    # Assert mlflow.set_experiment was called with the correct experiment name
    mock_set_experiment.assert_called_once_with("Accident Experiment")

    # Assert subprocess.run was called with the correct command
    mock_subprocess_run.assert_called_once_with(
        [
            "python", "mlflow_data/train.py",
            "--data", "data/dataset_Cramer.parquet",
            "--model_type", "KNN",
            "--n_neighbors", "5",
            "--weights", "distance"
        ],
        check=True
    )

@patch("mlflow_data.experiment.mlflow.set_experiment")
@patch("mlflow_data.experiment.subprocess.run")
def test_run_experiment_random_forest(mock_subprocess_run, mock_set_experiment):
    mock_subprocess_run.return_value = MagicMock()
    mock_set_experiment.return_value = None

    run_experiment(
        data_path="data/dataset_Cramer.parquet",
        model_type="RandomForest",
        n_estimators=10,
        max_depth=5
    )

    mock_set_experiment.assert_called_once_with("Accident Experiment")

    mock_subprocess_run.assert_called_once_with(
        [
            "python", "mlflow_data/train.py",
            "--data", "data/dataset_Cramer.parquet",
            "--model_type", "RandomForest",
            "--n_estimators", "10",
            "--max_depth", "5"
        ],
        check=True
    )

@patch("mlflow_data.experiment.mlflow.set_experiment")
@patch("mlflow_data.experiment.subprocess.run")
def test_run_experiment_stacking(mock_subprocess_run, mock_set_experiment):
    mock_subprocess_run.return_value = MagicMock()
    mock_set_experiment.return_value = None

    run_experiment(
        data_path="data/dataset_Cramer.parquet",
        model_type="Stacking"
    )

    mock_set_experiment.assert_called_once_with("Accident Experiment")

    mock_subprocess_run.assert_called_once_with(
        [
            "python", "mlflow_data/train.py",
            "--data", "data/dataset_Cramer.parquet",
            "--model_type", "Stacking"
        ],
        check=True
    )

@patch("mlflow_data.experiment.mlflow.set_experiment")
@patch("mlflow_data.experiment.subprocess.run")
def test_run_experiment_invalid_model(mock_subprocess_run, mock_set_experiment):
    with pytest.raises(ValueError, match="Invalid model type: InvalidModel. Expected one of \['KNN', 'RandomForest', 'Stacking'\]."):
        run_experiment(data_path="data/dataset_Cramer.parquet", model_type="InvalidModel")

    mock_subprocess_run.assert_not_called()
