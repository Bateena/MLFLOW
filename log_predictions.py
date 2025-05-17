import requests
import mlflow
import json

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Heart Disease Monitoring")

# Sample input 
inputs = [[
    52.0, 135.0, 212.0, 168.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 1.0
]]

# Send request
response = requests.post(
    url="http://127.0.0.1:1234/invocations",
    json={"inputs": inputs},
    headers={"Content-Type": "application/json"}
)

# Parse prediction
prediction = response.json()["predictions"][0]

# Log the input + prediction to MLflow
with mlflow.start_run(run_name="prediction_monitoring"):
    mlflow.log_param("input_row", json.dumps(inputs[0]))  # Save full input
    mlflow.log_metric("prediction", prediction)
    mlflow.set_tag("source", "inference_api")

print("Logged prediction:", prediction)
