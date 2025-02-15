import requests

BASE_URL = 'http://127.0.0.1:5001' 

def get_dummy_data():
    dummy_url = f"{BASE_URL}/generate_dummy"
    response = requests.get(dummy_url)
    if response.status_code == 200:
        dummy_data = response.json()
        print("\n✅ Dummy data generated:")
        print(dummy_data)  # Print first few keys
        return dummy_data
    else:
        print("\n❌ Failed to generate dummy data. Status:", response.status_code)
        print(response.text)
        return None

def test_predict(dummy_data):
    predict_url = f"{BASE_URL}/predict"
    response = requests.post(predict_url, json=dummy_data)
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Prediction result:")
        print(result)
    else:
        print("\n❌ Prediction request failed. Status:", response.status_code)
        print(response.text)

if __name__ == '__main__':
    dummy_data = get_dummy_data()
    if dummy_data:
        test_predict(dummy_data)
