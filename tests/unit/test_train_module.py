# tests/unit/test_train_module.py
import pytest
from endpoints.train import TrainParams, train_model

@pytest.fixture
def train_params():
    return TrainParams(
        data="data/dataset_Cramer.parquet",
        model_type="RandomForest",
        n_estimators=10,
        max_depth=5
    )

def test_train_model(train_params):
    result = train_model(train_params)
    assert "output" in result
    assert "Model RandomForest training started successfully" in result["message"]