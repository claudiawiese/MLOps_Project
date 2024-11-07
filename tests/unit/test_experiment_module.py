# tests/unit/test_experiment.py
import pytest
from accident_project.experiment import run_experiment

def test_run_experiment_knn():
    result = run_experiment(
        data_path="data/dataset_Cramer.parquet",
        model_type="KNN",
        n_neighbors=5,
        weights="distance"
    )
    assert "accuracy" in result
    assert result["accuracy"] > 0.5

def test_run_experiment_random_forest():
    result = run_experiment(
        data_path="data/dataset_Cramer.parquet",
        model_type="RandomForest",
        n_estimators=10,
        max_depth=5
    )
    assert "accuracy" in result
    assert result["accuracy"] > 0.5

def test_run_experiment_stacking():
    result = run_experiment(
        data_path="data/dataset_Cramer.parquet",
        model_type="Stacking",
    )
    assert "accuracy" in result
    assert result["accuracy"] > 0.5

def test_run_experiment_invalid_model():
    with pytest.raises(ValueError):
        run_experiment(data_path="data/dataset_Cramer.parquet", model_type="InvalidModel")
