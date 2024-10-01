# Imports librairies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Import Database
data = pd.read_csv("data/acc_data_encoded_2018-2022.parquet")
target = pd.read_csv("data/acc_target_encoded_2018-2022.parquet")

# Preprocessing Step
    # clean data
    # drop columns you need to drop
    # regroup columns if needed 

#Df
X = data # Drop target column
y = target # Define target column
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model 

#Define params, replace by your params 
params = {
    k = 5
    weights = 'distance'
}

#KNN model 
knn = KNeighborsClassifier(**params)
knn.fit(X_train, y_train)
     
# Here evaluate your model 
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
     
print(confusion_matrix(y_test, y_pred))