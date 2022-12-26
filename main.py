import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import tempfile

from video_extraction import *

st.title("Video Extraction Tool")
video = st.file_uploader("Upload your mp4 video.", ["mp4",])
st.video(video)


with st.form("File uploaded and video extraction"):
    algorithm = st.selectbox(label = "Algorithm", options=["SCIPY SVD","POWER ITERATION"])
    interval = st.slider("Interval",min_value = 20, max_value= 40)
    if video is not None: # run only when user uploads video
        vid = video.name
        with open(vid, mode='wb') as f:
            f.write(video.read()) 
    if st.form_submit_button("Extraction") and video is not None:
        with st.spinner("Waite for a while. The algorithm is extracting the background."):
            if algorithm == "SCIPY SVD":
                st.image(extract_bg(vid, interval = interval),use_column_width=True)
            elif algorithm == "POWER ITERATION":
                st.image(extract_bg_(vid, interval = interval),use_column_width=True)



