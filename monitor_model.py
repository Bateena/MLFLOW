import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("HeartDiseaseTrain-Test.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_encoded = pd.get_dummies(X, drop_first=True)

# Simulate new batch
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_test_scaled = scaler.fit(X_test).transform(X_test)

# Load model from registry
mlflow.set_tracking_uri("http://127.0.0.1:8080")  # 🔑 this is required
model_uri = "models:/HeartDiseaseModel/Production"
model = mlflow.sklearn.load_model(model_uri)

# Predict
y_pred = model.predict(X_test_scaled)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# Log as a monitoring run
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Heart Disease Monitoring")

with mlflow.start_run(run_name="monitoring"):
    mlflow.log_metrics(metrics)
    mlflow.set_tag("monitoring_type", "drift_check")
    print("Logged monitoring metrics:", metrics)
