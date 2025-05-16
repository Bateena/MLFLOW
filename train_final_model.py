import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")
X = pd.get_dummies(df.drop("target", axis=1), drop_first=True)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Final Heart Disease Model")

# Best parameters from tuning
params = {
    "C": 0.1480378690663537,
    "solver": "liblinear",
    "max_iter": 500
}

with mlflow.start_run(run_name="final_model") as run:
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Log performance
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    })

    # Register model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="heart_model_final",
        registered_model_name="HeartDiseaseModel"
    )

print(f"Model logged and registered: {model_info.model_uri}")
