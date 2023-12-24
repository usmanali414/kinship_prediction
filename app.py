from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras_vggface.utils import preprocess_input
import cv2
import numpy as np
import uvicorn
app = FastAPI()

# Load your Keras model
model = load_model("vgg_face_modelv1.h5")  # Replace with the path to your Keras model file

# Function to read and preprocess an image
def read_and_preprocess_image(file):
    image = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize the image to match your model's input size
    img = np.array(image).astype(np.float)
    image = preprocess_input(img, version=2)
    return image

@app.post("/predict/")
async def predict_images(file1: UploadFile, file2: UploadFile):
    #try:
    # Read and preprocess the uploaded images
    image1 = read_and_preprocess_image(file1)
    image2 = read_and_preprocess_image(file2)

    # Perform predictions using your Keras model
    prediction = model.predict([np.array([image1]), np.array([image2])])

    # Assuming a binary classification, you can extract the probability
    probability = round(prediction[0][0]*100,2)

    return JSONResponse(content={"Kinship Similarity": probability}, status_code=200)

    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
 
    uvicorn.run(app, host="0.0.0.0", port=8000)
