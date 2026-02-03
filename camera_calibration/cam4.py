import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('data/*.jpg')

fobjpoints = []
imgpoints = []
gray = None

for fname in images:
    gray = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Could not read {fname}")
        continue

    gray_blur = cv.GaussianBlur(gray, (5, 5), 0)

    ret, corners = cv.findChessboardCorners(
        gray_blur,
        (7, 6),
        flags=cv.CALIB_CB_ADAPTIVE_THRESH +
        cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        print(f"Found chessboard in {fname}")
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        imgpoints.append(corners2)

        vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(vis, (7, 6), corners2, ret)
        cv.imshow('img', vis)
        cv.waitKey(1000)
    else:
        print(f"❌ Chessboard NOT found in {fname}")

cv.destroyAllWindows()
# return objpoints, imgpoints, gray


# cv.destroyAllWindows()
