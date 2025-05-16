import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Encode categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)  # e.g., 'Sex' => 'Sex_Male'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Heart Disease Prediction")

# Define model parameters and train
params = {"solver": "liblinear", "C": 1.0, "max_iter": 100}
model = LogisticRegression(**params)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log with MLflow
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })
    mlflow.set_tag("model", "Logistic Regression for Heart Disease")
    mlflow.sklearn.log_model(model, artifact_path="heart_model", registered_model_name="HeartDiseaseModel")
