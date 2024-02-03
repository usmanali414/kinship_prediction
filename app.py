from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

import shutil
import os
import logging
import uuid

from utils.deepface_app import get_deepface_similarity
from utils.landmarks_app import get_landmarks_score
from utils.keras_model import read_and_preprocess_image, get_keras_model_prediction


logging.basicConfig(level=logging.INFO)

app = FastAPI()


def read_and_save_image(file: UploadFile, filename: str):
    # Generate a unique identifier for the image file
    unique_filename = str(uuid.uuid4()) + "_" + filename

    temp_dir = "temp_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with open(f"{temp_dir}/{unique_filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return os.path.join(temp_dir, unique_filename)


@app.post("/predict/")
async def predict_images(file1: UploadFile, file2: UploadFile):
    try:
        # Save the uploaded images to the temporary directory
        image_path1 = read_and_save_image(file1, file1.filename)
        image_path2 = read_and_save_image(file2, file2.filename)
        
        landmarks_score = get_landmarks_score(image_path1, image_path2)
        
        # Perform predictions using DeepFace
        deepface_score = get_deepface_similarity(image_path1, image_path2)
        # print("DeepFace Similarity Score:", deepface_score)

        # Read and preprocess the uploaded images separately for Keras model
        image1 = read_and_preprocess_image(image_path1)
        image2 = read_and_preprocess_image(image_path2)

        # Perform predictions using your Keras model
        model_score = get_keras_model_prediction(image1, image2)
        # print("Keras Model Probability:", model_score)
        if landmarks_score:
            prediciton =(deepface_score + model_score + landmarks_score) / 3
            logging.info("Average of landmarks score, DeepFace score, and model score calculated.")
        else:
            if deepface_score:
                prediciton =(deepface_score + model_score ) / 2
                logging.info("Average of DeepFace score and model score calculated.")
            else:
                prediciton = model_score
                logging.info("Model score used for prediction.")


        return JSONResponse(content={"Kinship Similarity": round(prediciton*100,2)}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Additional checks before removing files
        if os.path.exists(image_path1):
            os.remove(image_path1)
        if os.path.exists(image_path2):
            os.remove(image_path2)
        

if __name__ == "__main__":
 
    uvicorn.run(app, host="0.0.0.0", port=8000)
