# Imports librairies
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Import Database
data = pd.read_csv("raw_data_car_accidents.csv") #Rename according to csv

# Preprocessing Step
    # clean data
    # drop columns you need to drop
    # regroup columns if needed 

#Df
X = data.drop(columns=["target_column"]) # Drop target column
y = data["target_column"] # Define target column
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model 

#Define params, replace by your params 
params = {
    "n_estimators": 10,
    "max_depth": 10,
    "random_state": 42,
}

# Here put your model 
rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

# Here evaluate your model 
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

print(metrics)
