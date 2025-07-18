from google.cloud import vision

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

image_upload = st.file_uploader("Upload an Image", type=["jpeg", "jpg", "png"])

if image_upload:
    client = vision.ImageAnnotatorClient()  

    # read the image content from streamlit file uploader 
    content = image_upload.read()
    image = vision.Image(content = content)

    # OCR request
    response = client.text_detection(image=image)
    texts = response.text_annotations


    if texts:
        st.write("Detected Text")
        st.write(texts[0].description)
    else:
        st.write("No Text Found")
