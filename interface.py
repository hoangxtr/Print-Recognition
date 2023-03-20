"""Interface Modules"""
from __future__ import annotations

import streamlit as st
import numpy as np
import cv2

from utils import ocr


def main():
    st.title("Text Recognition Module")
    task = side_bar_nav()
    if task == "Automate Recognize Text":
        automate_ocr()


def side_bar_nav():
    with st.sidebar:
        module_selection = st.radio(
            "Choose module you want to use",
            ("Automate Recognize Text", "Add-on / Update Library", "Export Model / Library")
        )

    return module_selection

def automate_ocr():
    image = st.file_uploader("Upload an image to scan",
                             type = ["tif", "png", "jpg", "jpeg"],
                             accept_multiple_files= True)

    st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)

    if image:
        for image_value in image:
            filename = image_value.name
            file_bytes = np.asarray(bytearray(image_value.read()), dtype=np.uint8)
            st.write("Filename: ", image_value.name)
            st.write(ocr(filename))
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, channels="BGR")

if __name__ == "__main__":
    main()
