# Camera Calibration

## What does this do?

This script calibrates your camera using chessboard images. It detects chessboard corners in images and calculates camera calibration parameters (matrix and distortion coefficients) needed to correct lens distortion.

## Installation

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to use

Run the script from the `camera_calibration/` folder:
python calibration.py

The script will:

- Find all .jpg images in the chessboard_imgs folder
- Detect chessboard corners in each image
- Display the detected corners
- Calculate calibration parameters (saved in the script)

**Note:** Make sure your images contain clear chessboard patterns with 7x7 inner corners.
