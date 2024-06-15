# Testing the /test-predict-emotion route
import requests

test_url = 'http://localhost:5000/test-predict-emotion'
response = requests.post(test_url)
print(response.json())