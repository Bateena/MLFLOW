import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd

# Load and prepare data
df = pd.read_csv("HeartDiseaseTrain-Test.csv")
X = pd.get_dummies(df.drop("target", axis=1), drop_first=True)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Hyperopt Heart Tuning")

def objective(params):
    with mlflow.start_run(nested=True):
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        score = f1_score(y_test, preds)

        # Log params and score
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", score)

        return {'loss': -score, 'status': STATUS_OK}

# Search space
space = {
    'C': hp.uniform('C', 0.001, 10),
    'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
    'max_iter': hp.choice('max_iter', [100, 300, 500, 1000])
}

# Run optimization
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=Trials()
)

print("Best hyperparameters:", best)
