import requests
import numpy as np
import streamlit as st
import os
import sys
from PIL import Image
from io import BytesIO
sys.path.append('src')
# Determine the API URL based on the execution environment
if "DOCKER_MODE" in os.environ:
    api_url = "http://api:5000"
else:
    api_url = "http://localhost:5000"

st.title("TikTok dancing segmentation")
st.write('Upload your file!')
# Select the input type: video, image
input_type = st.selectbox("Select the input type", ("Image", "Video"))

if input_type == "Image":
    # Upload the input image file
    input_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if input_image is not None:
        # Process the image and get the output bytes
        image = Image.open(input_image)
        bytes_image = BytesIO()
        image.save(bytes_image, format="JPEG")
        bytes_image = bytes_image.getvalue()
        im = st.image(image, use_column_width=True)
        if st.button('Run'):
            im.empty()
            t = st.empty()
            t.markdown('Running...')
            predicted = requests.post(f"{api_url}/process_image", files={'file': bytes_image})
            predicted = predicted.json()
            predicted = np.array(predicted['prediction'])
            t.empty()
            t.markdown('Your prediction:')
            st.image(predicted, use_column_width=True)

elif input_type == "Video":
    input_video = st.file_uploader("Upload a video file", type=["mp4", "rb", "webm", 'avi'])
    if input_video is not None:
        video = input_video.getvalue()
        v = st.video(video)
        if st.button('Run'):
            v.empty()
            t = st.empty()
            t.markdown('Running...')
            predicted = requests.post(f"{api_url}/process_video", files={'file': input_video})
            predicted = predicted.content
            st.video(predicted)
