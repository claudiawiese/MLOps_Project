import argparse
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to the dataset file")
parser.add_argument("--model_type", type=str, required=True, help="Type of model: KNN, RandomForest, Stacking")
parser.add_argument("--n_neighbors", type=int, help="Number of neighbors for KNN")
parser.add_argument("--weights", type=str, help="Weight function for KNN")
parser.add_argument("--n_estimators", type=int, help="Number of trees for RandomForest")
parser.add_argument("--max_depth", type=int, help="Max depth for RandomForest")
parser.add_argument("--retrain", action="store_true", help="Retrain the model instead of training from scratch")
args = parser.parse_args()

# Load data
def load_data(data_path):
    # Load data based on file extension
    file_extension = os.path.splitext(data_path)[1].lower()  # Get file extension

    if file_extension == ".csv":
        df = pd.read_csv(data_path)
    elif file_extension == ".parquet":
        df = pd.read_parquet(data)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")

df = load_data(args.data)
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
experiment_name = "Accident Experiment"
mlflow.set_experiment(experiment_name)

# Define a unique run name (e.g., using model type and timestamp)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.model_type}_{timestamp}_run"

with mlflow.start_run(run_name=run_name) as run:
    # Initialize the model based on model type
    if args.model_type == "KNN":
        mlflow.log_param("model_type", "KNN")
        mlflow.log_param("n_neighbors", args.n_neighbors)
        mlflow.log_param("weights", args.weights)
        model = KNeighborsClassifier(n_neighbors=args.n_neighbors, weights=args.weights)
    elif args.model_type == "RandomForest":
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    elif args.model_type == "Stacking":
        mlflow.log_param("model_type", "Stacking")
        knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, weights=args.weights)
        rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
        model = VotingClassifier(estimators=[('knn', knn), ('rf', rf)], voting='soft')
    else:
        raise ValueError("Invalid model type specified")
    
    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics on the test set
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    classification_rep = classification_report(y_test, y_pred)

    # Log parameters and metrics to MLflow
    params = {
        "model_type": args.model_type,
        "n_neighbors": args.n_neighbors if args.model_type == "KNN" else None,
        "weights": args.weights if args.model_type == "KNN" else None,
        "n_estimators": args.n_estimators if args.model_type == "RandomForest" else None,
        "max_depth": args.max_depth if args.model_type == "RandomForest" else None
    }
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save the classification report to a text file and log it as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(classification_rep)
    mlflow.log_artifact("classification_report.txt")

    # Log the model itself to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=f"{args.model_type}_Accident_Model"  # Example of model name
    )
