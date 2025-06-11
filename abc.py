import requests

url = "http://127.0.0.1:8000/api"
data = {"question": "What is AIPIPE?", "api_key": "your-api-key"}

response = requests.post(url, json=data)
print(response.text)