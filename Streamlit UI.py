import streamlit as st
import requests
from PIL import Image

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Segment"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)
        st.write("Segmentation Result:", response.json()["prediction"])
