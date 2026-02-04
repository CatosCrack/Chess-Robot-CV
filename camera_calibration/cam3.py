import numpy as np
import cv2 as cv
import time
import pandas as pd
import glob

image = cv.imread('data/left01.jpg')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Better preprocessing for color images
gaussain_blur = cv.GaussianBlur(gray_image, (7, 7), 1)
gaussain_blur = cv.GaussianBlur(gaussain_blur, (7, 7), 1)

# Improve contrast with CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_img = clahe.apply(gaussain_blur)

ret, otsu_binary = cv.threshold(
    contrast_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('Otsu Binary', otsu_binary)

canny = cv.Canny(otsu_binary, 50, 150)
cv.imshow('Canny Edges', canny)


kernel = np.ones((5, 5), np.uint8)

img_dilation = cv.dilate(canny, kernel, iterations=2)
img_dilation = cv.erode(img_dilation, kernel, iterations=1)

cv.imshow('Dilated Image', img_dilation)
time.sleep(1)

lines = cv.HoughLinesP(img_dilation, 1, np.pi/180,
                       threshold=200, minLineLength=100, maxLineGap=50)

if lines is not None:
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]

        # draw lines
        cv.line(img_dilation, (x1, y1), (x2, y2), (255, 255, 255), 2)

cv.imshow('Lines Detected', img_dilation)
time.sleep(1)

board_contours, hierarchy = cv.findContours(
    img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

square_centers = list()

# draw filtered rectangles to "canny" image for better visualization
board_squared = canny.copy()

for contour in board_contours:
    if 4000 < cv.contourArea(contour) < 20000:
        # Approximate the contour to a simpler shape
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Ensure the approximated contour has 4 points (quadrilateral)
        if len(approx) == 4:
            pts = [pt[0] for pt in approx]  # Extract coordinates

            # Define the points explicitly
            pt1 = tuple(pts[0])
            pt2 = tuple(pts[1])
            pt4 = tuple(pts[2])
            pt3 = tuple(pts[3])

            x, y, w, h = cv.boundingRect(contour)
            center_x = (x+(x+w))/2
            center_y = (y+(y+h))/2

            square_centers.append([center_x, center_y, pt2, pt1, pt3, pt4])

            # Draw the lines between the points
            cv.line(board_squared, pt1, pt2, (255, 255, 0), 7)
            cv.line(board_squared, pt1, pt3, (255, 255, 0), 7)
            cv.line(board_squared, pt2, pt4, (255, 255, 0), 7)
            cv.line(board_squared, pt3, pt4, (255, 255, 0), 7)


cv.imshow('Board with Squares', board_squared)
time.sleep(2)

sorted_coordinates = sorted(square_centers, key=lambda x: x[1], reverse=True)

groups = []
current_group = [sorted_coordinates[0]]

for coord in sorted_coordinates[1:]:
    if abs(coord[1] - current_group[-1][1]) < 50:
        current_group.append(coord)
    else:
        groups.append(current_group)
        current_group = [coord]

# Append the last group
groups.append(current_group)

# Step 2: Sort each group by the second index (column values)
for group in groups:
    group.sort(key=lambda x: x[0])

# Step 3: Combine the groups back together
sorted_coordinates = [coord for group in groups for coord in group]

sorted_coordinates[:10]

cv.imshow('Final Board Squares', board_squared)
time.sleep(2)

cv.waitKey(5000)
cv.destroyAllWindows()
