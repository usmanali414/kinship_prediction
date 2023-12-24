import requests

# Replace with the URL where your FastAPI app is running
api_url = "http://localhost:8000/predict/"

# Replace with the file paths of your two images
image1_path = "P00001_face2.jpg"
image2_path = "P00003_face2.jpg"

# Create a dictionary to hold the files to upload
files = {
    "file1": ("image1.jpg", open(image1_path, "rb")),
    "file2": ("image2.jpg", open(image2_path, "rb")),
}

try:
    # Send a POST request to the FastAPI endpoint with the image files
    response = requests.post(api_url, files=files)

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        data = response.json()
        probability = data.get("probability")
        print(f"Probability: {data}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"An error occurred: {str(e)}")
