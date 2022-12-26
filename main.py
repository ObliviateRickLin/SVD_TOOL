import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import tempfile

from video_extraction import *

st.title("Video Extraction Tool")
with st.expander("Important! Click this line to read the instruction!"):
    st.write("""This is a sample web app for our course project. 
            You can choose scipy algorithm or our algorithm to extract the background. 
            You can adjust the interval from 20 to 40 by yourself.
            Note: our platform supports all the mp4 files in the test video files except the "street.mp4" one since it's too large
            to be stored on the disk of this free cloud platform. If this system crashes, email 120090527@link.cuhk.edu.cn. """)
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



