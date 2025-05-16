import mlflow

# Connect to the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

client = mlflow.tracking.MlflowClient()

# Replace with actual version number
model_name = "HeartDiseaseModel"
version = 2  # Change this if you registered multiple versions

# Move model to 'Production'
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model {model_name} version {version} transitioned to PRODUCTION.")
