### Accident Prediction API

### Launch docker for api and mflow 

docker-compose up --build 

api available at port 8000 and mlflow at port 5000 

To see endpoint you have access to Swagger 

    localhost:8000/docs 


### Endpoints

#### generate jwt token 

Example for admin

    curl -X POST "http://127.0.0.1:8000/token" -d "username=admin&password=admin123"

Example Response

    {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...your_token_here...",
        "token_type": "bearer"
    }

#### train endpoint 

    curl -X POST "http://127.0.0.1:8000/train" -H "Content-Type: application/json" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...your_token_here..." -d '{
    "data": "data/dataset_Cramer.parquet",
    "model_type": "KNN",
    "n_neighbors": 5,
    "weights": "distance",
    "n_estimators": 0,
    "max_depth":0,
    "retrain": false
    }'

Available models to train: KNN, RandomForest, Stacking with KNN and RandomForest

#### retrain endpoint 

    To be defined 

#### predict endpoint 

    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -H "Authorization: Bearer  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...your_token_here..." -d '{
    "model_uri": "models:/KNN_Accident_Model/1",
    "data_path": "data/sample_data_for_prediction.parquet"
    }'

#### predict_with_pretrained_model endpoint

curl -X 'POST' \
  'http://localhost:8000/predict_with_pretrained_model' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTczMTE1NTc3MH0.RdJHj4ZPcGtTIUu5LGvjuDns2Dy7tzurjasIRfz3Q84' \
  -H 'Content-Type: application/json' \
  -d '{
  "data_path": "data/sample_data_for_prediction.parquet",
  "pretrained_model_path": "pretrained_models/Stacking_RF_cv5.joblib"
}'
