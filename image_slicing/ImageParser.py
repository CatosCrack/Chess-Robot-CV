"""
ImageParser: Label images from images_to_parse one by one (Empty=0, Filled=1)
and collect results in an Array for use with Array.csvExport().
"""
import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

# is this the fking thing thatt causes TypeError???
# relook this over
sys.path.insert(0, os.path.expanduser("~/Downloads"))
from Array import Array

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_TO_PARSE = os.path.join(SCRIPT_DIR, "images_to_parse")
PARSED_IMAGES = os.path.join(SCRIPT_DIR, "parsed_images")

# Ensure destination exists
os.makedirs(PARSED_IMAGES, exist_ok=True)


def load_image_paths():
    exts = (".png", ".jpg", ".jpeg")
    if not os.path.isdir(IMAGES_TO_PARSE):
        return []
    return sorted(
        f for f in os.listdir(IMAGES_TO_PARSE)
        if f.lower().endswith(exts)
    )


def label_and_move(filename: str, label: int, array: Array) -> bool:
    src = os.path.join(IMAGES_TO_PARSE, filename)
    new_name = f"{label}_{filename}"
    dest = os.path.join(PARSED_IMAGES, new_name)
    if not os.path.isfile(src):
        return False
    try:
        os.rename(src, dest)
        return array.attachToArray((dest, label))
    except Exception as e:
        st.error(f"Error: {e}")
        return False


def main():
    st.set_page_config(page_title="Image Parser", layout="centered")
    st.title("Image Parser")
    st.caption("Label each image as Empty (0) or Filled (1). Results are stored in an Array for use with Array.csvExport().")

    # Session state: current index and shared Array
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "array" not in st.session_state:
        # Array.__init__ in Array.py returns self.df, which raises TypeError; avoid calling __init__
        arr = Array.__new__(Array)
        arr.df = pd.DataFrame(columns=["filepath", "label"])
        st.session_state.array = arr

    image_files = load_image_paths()
    array = st.session_state.array
    idx = st.session_state.current_index

    if not image_files:
        st.info('No images in "images_to_parse". Add PNG/JPG/JPEG files and refresh.')
        return

    # Bounds: only show current image while there are unprocessed ones
    if idx >= len(image_files):
        st.success("All images labeled.")
        st.dataframe(array.df, use_container_width=True, hide_index=True)
        if st.button("Export to CSV"):
            if array.csvExport():
                st.success("Saved to data/dataset.csv")
            else:
                st.error("Export failed.")
        return

    filename = image_files[idx]
    filepath = os.path.join(IMAGES_TO_PARSE, filename)

    st.write(f"**Image {idx + 1} of {len(image_files)}** — `{filename}`")
    try:
        img = Image.open(filepath)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Empty (0)", key="empty"):
            if label_and_move(filename, 0, array):
                st.session_state.current_index = idx + 1
                st.rerun()
    with col2:
        if st.button("Filled (1)", key="filled"):
            if label_and_move(filename, 1, array):
                st.session_state.current_index = idx + 1
                st.rerun()

    # Optional: show progress and current dataframe
    with st.expander("Progress & data"):
        st.write(f"Labeled so far: {len(array.df)}")
        if len(array.df) > 0:
            st.dataframe(array.df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
