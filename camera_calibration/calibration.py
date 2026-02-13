import numpy as np
import cv2 as cv
import glob

# Chessboard size (INNER corners)
CHECKERBOARD = (7, 7)

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
        # append a copy of objp so each view has its own array
        objpoints.append(objp.copy())

        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('Corners', img)
        cv.waitKey(500)

cv.destroyAllWindows()

#debug
if not objpoints:
    raise RuntimeError("No chessboard corners were found in any image. "
                       "Check your 'data/*.jpg' files and CHECKERBOARD size.")


# === CALIBRATION (ONCE) ===
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

#=== UNDISTORT ALL IMAGES ===

for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist,(w, h), 1, (w,h)
    )
    dst = cv.undistort(img,mtx,dist,None,newcameramtx)
    #crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('Undistorted', dst)
    cv.waitKey(0)  # press any key to advance to the next image
cv.destroyAllWindows()


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))