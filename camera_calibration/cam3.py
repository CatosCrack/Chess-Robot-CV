import numpy as np
import cv2 as cv
import glob

# Chessboard size (INNER corners)
CHECKERBOARD = (7, 6)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('data/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('Corners', img)
        cv.waitKey(200)

cv.destroyAllWindows()

# === CALIBRATION (ONCE) ===
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# === UNDISTORT ONE IMAGE ===
img = cv.imread(images[0])
h, w = img.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(
    mtx, dist, (w, h), 1, (w, h)
)

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv.imwrite('calibresult.png', dst)
print("Saved undistorted image as calibresult.png")
