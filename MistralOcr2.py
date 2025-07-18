import os
from mistralai import Mistral
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")

client = Mistral(api_key=api_key)

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpeg", "jpg"])
if uploaded_image:
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
        "type": "document_bytes",
        "bytes":uploaded_image.read(),  # Read image bytes from uploaded file
        "mime":uploaded_image.type,  #e.g  "image/png"
    },
    )

    st.write(ocr_response)