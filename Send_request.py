import requests

# Format for MLflow 2.x and above
data = {
    "inputs": [[
        52.0, 125.0, 212.0, 168.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 1.0
    ]]
}

response = requests.post(
    url="http://127.0.0.1:1234/invocations",
    json=data,
    headers={"Content-Type": "application/json"}
)

print("Prediction:", response.json())
