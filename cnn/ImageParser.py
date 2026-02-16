"""
ImageParser: Label images from images_to_parse one by one (Empty=0, Filled=1)
and collect results in an Array for use with Array.csvExport().
"""
import os
import sys
import pandas as pd
import streamlit as st
from PIL import Image
from Array import Array
from pathlib import Path

# Define paths relative to this script and ensure destination folder exists
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_TO_PARSE = os.path.join(SCRIPT_DIR, "data/images_to_parse")
PARSED_IMAGES = os.path.join(SCRIPT_DIR, "data/parsed_images")
os.makedirs(PARSED_IMAGES, exist_ok=True)

# Returns a sorted lists of image absolute paths in images_to_parse with valid extensions
def load_image_paths() -> list:
    exts = (".png", ".jpg", ".jpeg")
    path_obj = Path(IMAGES_TO_PARSE)
    if not path_obj.is_dir():
        return []
    return sorted(
        file for file in path_obj.iterdir()
        if file.suffix.lower() in exts
    )

# Takes a filename and label, moves the processed image to 
# parsed_images and appends the filename and label to the dataset
def label_and_move(filepath_obj: Path, label: int, array: Array) -> bool:
    dest = Path(PARSED_IMAGES) / filepath_obj.name
    try:
        filepath_obj.rename(dest)
        # Store path relative to script dir so dataset is portable across machines
        rel_path = dest.relative_to(Path(SCRIPT_DIR))
        array.attachToArray((rel_path.as_posix(), label))
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# Define Streamlit app
def main():
    st.set_page_config(page_title="Image Parser", layout="centered")
    st.title("Image Parser")
    st.caption("Label each image as Empty (0) or Filled (1). Results are stored in an Array for use with Array.csvExport().")

    # Session state: current index and shared Array
    if "array" not in st.session_state:
        st.session_state.array = Array()

    image_file_list = load_image_paths()
    array = st.session_state.array

    # Bounds: only show current image while there are unprocessed ones
    if not image_file_list:
        st.success("All images labeled.")
        st.dataframe(array.df, use_container_width=True, hide_index=True)
        if st.button("Export to CSV"):
            if array.csvExport():
                st.success("Saved to data/dataset.csv")
            else:
                st.error("Export failed.")
        return

    filepath_obj = image_file_list[0]  # Always take the first unprocessed image

    try:
        img = Image.open(filepath_obj)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Empty (0)", key="empty"):
            if label_and_move(filepath_obj, 0, array):
                st.rerun()
    with col2:
        if st.button("Filled (1)", key="filled"):
            if label_and_move(filepath_obj, 1, array):
                st.rerun()

    # Optional: show progress and current dataframe
    with st.expander("Progress & data"):
        st.write(f"Labeled so far: {len(array.df)}")
        if len(array.df) > 0:
            st.dataframe(array.df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
