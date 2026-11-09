
import requests
import json

# API endpoint
url = "http://localhost:9696/predict"

# Input data
data = {
    "prolongued_decelerations": 0.002,
    "accelerations": 0.000,
    "uterine_contractions": 0.007,
    "light_decelerations": 0.006,
    "fetal_movement": 0.001,
    "severe_decelerations": 0.000,
    "baseline_value": 129.0
}

try:
    # Send POST request
    response = requests.post(url, json=data)

    # Check response status
    response.raise_for_status()

    # Display prediction result
    print("Prediction successful!")
    print(json.dumps(response.json(), indent=4))

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API. Make sure your server is running.")

except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
