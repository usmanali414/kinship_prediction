from deepface import DeepFace
import logging
def get_deepface_similarity(image_path1, image_path2):
    try:
        result = DeepFace.verify(image_path1, image_path2, enforce_detection=False)
        similarity_score = result["distance"]
        score = 1 - similarity_score 
        return score
    except Exception as e:
        logging.error(f"Could not calculate the DeepFace score: {e}")
        return None