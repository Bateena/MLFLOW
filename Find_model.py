import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")

run_id = "<d97b43a21ebe4f328782d761ad8cba26>"
model_uri = f"runs:/{run_id}/heart_model"

result = mlflow.register_model(
    model_uri=model_uri,
    name="HeartDiseaseModel"
)

print(f"Registered model version: {result.version}")
