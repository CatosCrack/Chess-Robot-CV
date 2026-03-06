#Image Slicing Role 1: "The Line Detective" (Masood Quraishi)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os



#Storing the Chess Board Image
img =cv2.imread("board1.jpg")
resize_img=cv2.resize(img,(860,860))  # Changing The resolution

rgb_image=cv2.cvtColor(resize_img,cv2.COLOR_BGR2RGB)     
gry_image=cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)

plt.imshow(rgb_image) #Plotting the image on a graph

# Gaussian Blur Image Coversion
gaussian_blur=cv2.GaussianBlur(gry_image,(11,11),0)
plt.imshow(gaussian_blur,cmap="gray") 

#OTSU Threshold (Converting the image into binary that is Black and White)
ret,otsu_binary=cv2.threshold(gaussian_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(otsu_binary,cmap="gray")

#Canny Edge Detection 
canny=cv2.Canny(otsu_binary,20,255)
plt.imshow(canny,cmap="gray")

#Dilation
dil=np.ones((7,7),np.uint8)
img_dilation=cv2.dilate(canny,dil,iterations=1)
plt.imshow(img_dilation,cmap="gray")

#Hough Lines
lines=cv2.HoughLinesP(img_dilation,1,np.pi/180,threshold=200,minLineLength=100,maxLineGap=50)
if lines is not None:
    for i, line in enumerate(lines):
        x1,y1,x2,y2=line[0]

        cv2.line(img_dilation,(x1,y1),(x2,y2),(255,255,255),2)

plt.imshow(img_dilation,cmap="gray")


#Dilation Again after Hough Lines
dil=np.ones((3,3),np.uint8)
img_dilation_2=cv2.dilate(img_dilation,dil,iterations=1)

plt.imshow(img_dilation_2,cmap="gray")


# Role 2: Neve Turner: Find and Filter Contours

#Find all the contours from the dilated image
board_contours, hierarchy = cv2.findContours(img_dilation_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# List will store center coordinates and corner points of detected squares
square_centers = list()

board_squared = canny.copy()

# Iterate through all detected contours
for contour in board_contours:
    if 4000 < cv2.contourArea(contour) < 20000:
        epsilon = 0.02 * cv2.arcLength(contour, True) 
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Keeping only square contours 
        if len(approx) == 4:
            pts = [pt[0] for pt in approx] # Extracting coordinates

            # Defining the points
            pt1 = tuple(pts[0])
            pt2 = tuple(pts[1])
            pt3 = tuple(pts[2])
            pt4 = tuple(pts[3])

            x, y, w, h = cv2.boundingRect(contour)

            # Calculating the center of the square
            center_x= x+w/2
            center_y= y+h/2

            # Storing square information in list
            square_centers.append([center_x, center_y, pt2, pt1, pt3, pt4])
            
            # Drawring the edges of the detected square 
            cv2.drawContours(board_squared, [approx], -1, (255, 255, 0), 7)

plt.imshow(board_squared, cmap="grey")

# Sorting coordinates
sorted_coordinates = sorted(square_centers, key=lambda x: x[1], reverse=True)

groups = []
current_group = [sorted_coordinates[0]]

for coord in sorted_coordinates[1:]:
    if abs(coord[1] - current_group[-1][1]) < 50:
        current_group.append(coord)
    else:
        groups.append(current_group)
        current_group = [coord]

groups.append(current_group) # Append the last group


for group in groups: # Sort each group by the second index (column values)
    group.sort(key=lambda x: x[0])


sorted_coordinates = [coord for group in groups for coord in group] # Combine the groups back together

# Finding undetected squares
for num in range(len(sorted_coordinates)-1): 
    if abs(sorted_coordinates[num][1] - sorted_coordinates[num+1][1])< 50 :
        if sorted_coordinates[num+1][0] - sorted_coordinates[num][0] > 150:
            x=(sorted_coordinates[num+1][0] + sorted_coordinates[num][0])/2
            y=(sorted_coordinates[num+1][1] + sorted_coordinates[num][1])/2
            p1=sorted_coordinates[num+1][5]
            p2=sorted_coordinates[num+1][4]
            p3=sorted_coordinates[num][3]
            p4=sorted_coordinates[num][2]
            sorted_coordinates.insert(num+1,[x,y,p1,p2,p3,p4])

num_squares = len(sorted_coordinates)
print(f"Total squares detected: {num_squares}")

# Group into rows of 8 dynamically
for i in range(0, num_squares, 8):
    row = sorted_coordinates[i : i + 8]
    # This prints the (x, y) center for each square in the row
    print(f"Row {i//8}: {[[round(c[0]), round(c[1])] for c in row]}")

board_vis = cv2.cvtColor(board_squared, cv2.COLOR_GRAY2BGR)
square_num=1
for cor in sorted_coordinates:
  cv2.putText(img = board_vis,text = str(square_num),org = (int(cor[0])-30, int(cor[1])),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1,color = (255, 246, 55),thickness = 1)
  square_num+=1

# Role 5: debug and giving feedback
debug_img = resize_img.copy()  

# Draw hough lines in red
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# Draw detected square contours in green
for contour in board_contours:
    if 4000 < cv2.contourArea(contour) < 20000:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 2)

# Draw square numbers in cyan
for i, coord in enumerate(sorted_coordinates):
    cx, cy = int(coord[0]), int(coord[1])
    cv2.circle(debug_img, (cx, cy), 5, (255, 255, 0), -1)
    cv2.putText(debug_img, str(i + 1), (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# Small legend in the corner
cv2.rectangle(debug_img, (5, 5), (230, 75), (0, 0, 0), -1)
cv2.putText(debug_img, "RED   = Hough Lines",    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(debug_img, "GREEN = Square Contours", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(debug_img, "CYAN  = Square Centers",  (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

cv2.imshow("Role 5 - Debug Overlay", debug_img)

# Print a test report 

total = len(sorted_coordinates)
print("\nTEST REPORT:")
print(f"Squares detected : {total} / 64")

if total == 64:
    print("Result           : PASS ✓")
elif total > 64:
    print("Result           : WARN — too many squares detected")
    print("Feedback         : Role 2 — tighten contour area filter (currently 4000-20000)")
else:
    print(f"Result           : WARN — {64 - total} squares missing")
    print("Feedback         : Role 1 — try lowering Canny low threshold (currently 20)")
    print("Feedback         : Role 2 — try widening contour area range")

if lines is None:
    print("Feedback         : Role 1 — No Hough lines found! Lower the threshold parameter")

# Role 4: The Data Architect (Samuel / Infrastructure)
# Saves every square as game_{id}_{move}_{square}.jpg and writes a JSON log.

def save_dataset(image, grid_matrix, output_folder, game_id="001", move="001"):
    squares_dir = os.path.join(output_folder, "squares")
    os.makedirs(squares_dir, exist_ok=True)

    log_entries = []
    square_idx = 1

    for row_idx, row in enumerate(grid_matrix):
        for col_idx, square in enumerate(row):
            center_x, center_y = square[0], square[1]
            pts = [square[2], square[3], square[4], square[5]]

            # Derive bounding box from the four corner points
            xs = [int(p[0]) for p in pts]
            ys = [int(p[1]) for p in pts]
            x  = max(0, min(xs))
            y  = max(0, min(ys))
            x2 = min(image.shape[1], max(xs))
            y2 = min(image.shape[0], max(ys))

            crop = image[y:y2, x:x2]

            filename = f"game_{game_id}_{move}_{square_idx:02d}.jpg"
            cv2.imwrite(os.path.join(squares_dir, filename), crop)

            log_entries.append({
                "square":  square_idx,
                "row":     row_idx,
                "col":     col_idx,
                "center":  [int(round(center_x)), int(round(center_y))],
                "bbox":    {"x": int(x), "y": int(y), "w": int(x2 - x), "h": int(y2 - y)},
                "status":  "saved" if crop.size > 0 else "empty",
                "file":    filename
            })

            square_idx += 1

    log = {"game_id": game_id, "move": move, "total_squares": square_idx - 1, "squares": log_entries}
    json_path = os.path.join(output_folder, f"game_{game_id}_{move}_log.json")
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nRole 4 — Saved {square_idx - 1} squares to '{squares_dir}'")
    print(f"Role 4 — JSON log written to '{json_path}'")


# Build 8x8 grid_matrix from sorted_coordinates and call save_dataset
if len(sorted_coordinates) == 64:
    grid_matrix = [sorted_coordinates[i:i+8] for i in range(0, 64, 8)]
    save_dataset(resize_img, grid_matrix, output_folder="chess_squares_output")
else:
    print(f"\nRole 4 — Skipped: need 64 squares, got {len(sorted_coordinates)}")


cv2.waitKey(0)
cv2.destroyAllWindows()
