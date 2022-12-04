import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import tempfile

from video_extraction import *

st.title("Video Extraction Tool")
video = st.file_uploader("Upload your mp4 video.", ["mp4",])
st.video(video)
if video:
    tpath = tempfile.NamedTemporaryFile(delete=False)
    tpath.write(video.read())
    st.image(extract_bg(tpath.name),use_column_width=True)




