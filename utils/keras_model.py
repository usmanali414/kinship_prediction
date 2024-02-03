from keras.models import load_model
from keras_vggface.utils import preprocess_input
import cv2
import numpy as np

# Load your Keras model
model = load_model("ts_kinship_vgg_face 1-101-ts.h5")  # Replace with the path to your Keras model file

# Function to read and preprocess an image
def read_and_preprocess_image(file):
    image = cv2.imread(file)
    # image = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize the image to match your model's input size
    img = np.array(image).astype(np.float64)
    image = preprocess_input(img, version=2)
    return image


def get_keras_model_prediction(image1, image2):
    prediction = model.predict([np.array([image1]), np.array([image2])])
    return prediction[0][0]