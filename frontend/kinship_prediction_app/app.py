import streamlit as st
import requests

st.set_page_config(
    page_title="Kinship Prediction",
    page_icon="🤴",


)

st.header('Kinship Prediction',anchor=False, divider='rainbow')

col1, col2 = st.columns(2)

with col1:
    # Upload images using file_uploader
    image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    # Display images if uploaded
    if image1 is not None:
        st.image(image1, width=200)


with col2:
    # Upload images using file_uploader
    image2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
    # Display images if uploaded
    if image2 is not None:
        st.image(image2, width=200)

# Display images if uploaded
if image1 is not None and image2 is not None:
    # Button to trigger image similarity prediction
    if st.button("Predict Similarity", type= 'primary'):
        # Create a dictionary to hold the files to upload
        files = {"file1": ("image1.jpg", image1), "file2": ("image2.jpg", image2)}

        # API endpoint URL
        api_url = "http://18.219.9.218:8000/predict/"




        try:
            # Send a POST request to the FastAPI endpoint with the image files
            response = requests.post(api_url, files=files)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                data = response.json()
                similarity_score = data.get("Kinship Similarity")
                
                st.success(f"Kinship Similarity: {similarity_score}")

            else:
                st.error(f"Request failed with status code: {response.status_code}")
                st.error(response.text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")












