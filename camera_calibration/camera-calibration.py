import numpy as np
import cv2 as cv
import glob



CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OBJP = np.zeros((6*7, 3), np.float32)
OBJP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)


def get_image_files(path):
    return glob.glob(path)


def find_image_points(images, criteria=CRITERIA, objp=OBJP):
  
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    gray = None

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners 
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    return objpoints, imgpoints, gray


def calibrate_camera(objpoints, imgpoints, gray):
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistort_and_save(mtx, dist, image_path='data/left12.jpg', out_file='calibresult.png'):
    img = cv.imread(image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite(out_file, dst)


def main():
    images = get_image_files('data/*.jpg')
    objpoints, imgpoints, gray = find_image_points(images)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray)

    undistort_and_save(mtx, dist, image_path='data/left12.jpg', out_file='calibresult.png')


if __name__ == '__main__':
    main()