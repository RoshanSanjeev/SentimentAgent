import requests

url = "https://roshansanjeev-sentimentanalysis.hf.space/"  # Your Hugging Face Space URL
payload = {
    "text": "I want to hurt others"
}

response = requests.post(url, json=payload)
print("Response:", response.json())


