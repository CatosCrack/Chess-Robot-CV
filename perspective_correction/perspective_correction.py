

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from pathlib import Path

# Configuration dictionary
CONFIG = {
    "image_path": r"perspective_correction/test_images/board1.jpg",
    "output_path": r"perspective_correction/output/board_with_grid.jpg",
    "hough_params": {
        "threshold": 120,
        "minLineLength": 100,
        "maxLineGap": 50,
    },
    "square_area_range": (2000, 20000),
    "square_length_threshold": 35,
    "board_size": (8, 8),
    "output_size": (1200, 1200),
}

# Batch processing configuration
CONFIG.update({
    "input_dir": r"perspective_correction/test_images",
    "output_dir": r"perspective_correction/output",
    "file_extensions": [".jpg", ".jpeg", ".png"],
    # Heuristics for using square-based detection vs fallback
    "min_squares_for_board": 20,
    "min_hull_area_ratio": 0.03,
    # Minimum warp score to accept a candidate (0-1). Lowered to allow
    # more flexible detection on occluded / reflective boards.
    "min_warp_score": 0.05,
})

def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image at {image_path}.")
    return image

def apply_threshold(image):
    """Apply OTSU thresholding to the grayscale image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary_image

def detect_edges(binary_image):
    """Detect edges using Canny edge detection."""
    v = np.median(binary_image)
    edges = cv2.Canny(binary_image, 0.66 * v, 1.33 * v)
    return edges

def detect_lines(edges, hough_params):
    """Detect lines using the Hough Line Transform."""
    kernel = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        dilated_edges,
        1,
        np.pi / 180,
        threshold=hough_params["threshold"],
        minLineLength=hough_params["minLineLength"],
        maxLineGap=hough_params["maxLineGap"],
    )
    return lines

def find_contours(lines, image_shape):
    """Find contours from the detected lines."""
    black_image = np.zeros(image_shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black_image, (x1, y1), (x2, y2), 255, 2)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(black_image, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def filter_squares(contours, area_range, length_threshold):
    """Filter contours to find valid squares."""
    valid_squares = []
    for contour in contours:
        # Ensure the contour is a NumPy array with the correct shape
        contour = np.array(contour, dtype=np.float32).reshape(-1, 1, 2)
        
        if len(contour) < 3:  # Skip invalid contours with fewer than 3 points
            continue

        area = cv2.contourArea(contour)
        if area_range[0] < area < area_range[1]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                pts = [pt[0].tolist() for pt in approx]
                lengths = [
                    math.sqrt((pts[i][0] - pts[(i + 1) % 4][0]) ** 2 +
                              (pts[i][1] - pts[(i + 1) % 4][1]) ** 2)
                    for i in range(4)
                ]
                if max(lengths) - min(lengths) <= length_threshold:
                    valid_squares.append(pts)
    return valid_squares

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left.

    Uses the summed and differenced coordinates method which is robust
    to typical perspective distortions.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # sum: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # diff: top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1).reshape(4)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def compute_and_validate_homography(src_pts: np.ndarray, dst_pts: np.ndarray, dst_size: tuple, src_image_shape: tuple,
                                    min_area_ratio: float = 0.002, max_cond: float = 1e8) -> tuple:
    """Compute homography and validate it.

    Returns (M, reason). M is None and reason non-empty on failure.
    """
    # Use a double-precision DLT with Hartley normalization for numerical stability
    def find_homography_dlt(src, dst):
        src = src.astype(np.float64)
        dst = dst.astype(np.float64)

        def normalize(pts):
            mean = pts.mean(axis=0)
            pts_centered = pts - mean
            avg_dist = np.sqrt((pts_centered ** 2).sum(axis=1)).mean()
            if avg_dist == 0:
                scale = 1.0
            else:
                scale = np.sqrt(2) / avg_dist
            T = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]], dtype=np.float64)
            pts_norm = (T @ np.vstack([pts.T, np.ones((pts.shape[0],))])).T[:, :2]
            return pts_norm, T

        src_n, Tsrc = normalize(src)
        dst_n, Tdst = normalize(dst)

        N = src_n.shape[0]
        A = np.zeros((2 * N, 9), dtype=np.float64)
        for i in range(N):
            x, y = src_n[i]
            u, v = dst_n[i]
            A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]
        Hn = h.reshape(3, 3)
        # Denormalize
        H = np.linalg.inv(Tdst) @ Hn @ Tsrc
        # Normalize so that H[2,2] == 1 if possible
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    # First try numerically-stable DLT
    try:
        M = find_homography_dlt(src_pts, dst_pts)
    except Exception as e:
        # Try a robust RANSAC fallback
        try:
            M_ransac, mask = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), cv2.RANSAC, 5.0)
            if M_ransac is None:
                return None, f"DLT homography error and RANSAC failed: {e}"
            M = M_ransac.astype(np.float64)
        except Exception as e2:
            return None, f"DLT homography error and RANSAC error: {e}; {e2}"

    if not np.isfinite(M).all():
        # Try RANSAC as a second attempt
        try:
            M_ransac, mask = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), cv2.RANSAC, 5.0)
            if M_ransac is None or not np.isfinite(M_ransac).all():
                return None, "Homography contains non-finite values"
            M = M_ransac.astype(np.float64)
        except Exception:
            return None, "Homography contains non-finite values"

    # Check condition number; if too large, try RANSAC fallback
    try:
        cond = np.linalg.cond(M)
    except Exception:
        cond = float('inf')
    if cond > max_cond:
        try:
            M_ransac, mask = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), cv2.RANSAC, 5.0)
            if M_ransac is None or not np.isfinite(M_ransac).all():
                return None, f"Homography condition number too large: {cond:.2e}"
            cond_r = np.linalg.cond(M_ransac.astype(np.float64))
            if cond_r > max_cond:
                return None, f"Homography condition number too large (DLT {cond:.2e}, RANSAC {cond_r:.2e})"
            M = M_ransac.astype(np.float64)
        except Exception:
            return None, f"Homography condition number too large: {cond:.2e}"

    # Project source polygon and check area in destination
    try:
        projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    except cv2.error as e:
        return None, f"perspectiveTransform error: {e}"

    area_dst = abs(cv2.contourArea(projected.reshape(-1, 1, 2)))
    dst_area = float(dst_size[0] * dst_size[1])
    if dst_area <= 0:
        return None, "Invalid destination size"
    if area_dst < min_area_ratio * dst_area:
        return None, f"Projected area too small: {area_dst:.1f} (< {min_area_ratio*dst_area:.1f})"

    # Also check source polygon area relative to original image
    area_src = abs(cv2.contourArea(src_pts.reshape(-1, 1, 2)))
    img_area = float(src_image_shape[0] * src_image_shape[1])
    if img_area > 0 and area_src < 1e-6 * img_area:
        return None, f"Source polygon area suspiciously small: {area_src:.1f}"

    return M, ""


def score_warped_image(warp: np.ndarray) -> float:
    """Score a warped image on how likely it contains an 8x8 checkerboard.

    Returns a float in [0, ~1], higher is better.
    """
    try:
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0

    h, w = gray.shape[:2]
    cw = w // 8
    ch = h // 8
    if cw < 4 or ch < 4:
        return 0.0

    crop_w = cw * 8
    crop_h = ch * 8
    gray_crop = gray[:crop_h, :crop_w]

    # Cell means
    cells = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            cell = gray_crop[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw]
            cells[i, j] = float(cell.mean())

    # Contrast between adjacent cells
    diff_h = np.abs(cells[:, :-1] - cells[:, 1:]).mean()
    diff_v = np.abs(cells[:-1, :] - cells[1:, :]).mean()
    cell_contrast = (diff_h + diff_v) / 2.0 / 255.0

    # Edge density inside crop
    edges = cv2.Canny(gray_crop, 50, 150)
    edge_density = float(edges.mean()) / 255.0

    # Alternating pattern score using sign pattern correlation
    sign = ((np.indices((8, 8)).sum(axis=0) % 2) * 2 - 1).astype(np.float32)
    std = cells.std()
    if std > 1e-6:
        cells_norm = (cells - cells.mean()) / std
        pattern_score = abs((cells_norm * sign).mean())
    else:
        pattern_score = 0.0

    # Weighted combination
    score = 0.55 * cell_contrast + 0.35 * edge_density + 0.10 * pattern_score
    return float(score)

def perspective_correction(image, squares, output_size):
    """Apply perspective correction to the image using all detected squares.

    Build a convex hull around all square corners and approximate a quadrilateral
    representing the board. If approximation fails, fall back to `minAreaRect`.
    """
    if not squares:
        raise ValueError("No valid squares found for perspective correction.")

    # Flatten all square corner points into a single point set
    pts_all = np.array([pt for sq in squares for pt in sq], dtype=np.float32).reshape(-1, 2)

    # Compute convex hull of all detected points
    hull = cv2.convexHull(pts_all).reshape(-1, 2)

    # Attempt to approximate the hull with a 4-point polygon (quadrilateral)
    epsilon = 0.02 * cv2.arcLength(hull.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)

    if approx.shape[0] == 4:
        src_pts = approx.astype(np.float32)
    else:
        # Fallback: use minAreaRect to get a rotated rectangle around the point cloud
        rect = cv2.minAreaRect(pts_all)
        box = cv2.boxPoints(rect)
        src_pts = np.array(box, dtype=np.float32)

    # Order source points and build destination rectangle (tl,tr,br,bl)
    src_ordered = order_points(src_pts)
    # Try to refine the approximate quad using edge-fitted lines and
    # subpixel corner refinement for better precision.
    try:
        refined = refine_quad_via_edge_fits(src_ordered, image, strip_width=32)
        if refined is not None and refined.shape == (4, 2):
            src_ordered = order_points(refined)
    except Exception:
        pass
    width, height = output_size
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Validate homography before applying
    M, reason = compute_and_validate_homography(src_ordered, dst_pts, (width, height), image.shape)
    if M is None:
        raise ValueError(f"Invalid homography: {reason}")

    warped_image = cv2.warpPerspective(image, M, (width, height))
    return warped_image, M

def draw_chessboard_grid(image, board_size):
    """Draw an 8x8 chessboard grid on the image."""
    rows, cols = board_size
    height, width = image.shape[:2]
    square_width = width // cols
    square_height = height // rows
    for i in range(rows):
        for j in range(cols):
            top_left = (j * square_width, i * square_height)
            bottom_right = ((j + 1) * square_width, (i + 1) * square_height)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image


def detect_board_quad(image: np.ndarray) -> np.ndarray:
    """Detect a 4-point quadrilateral surrounding the board.

    Returns a (4,2) float32 ndarray in arbitrary order, or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(th, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if approx.shape[0] >= 4:
        hull = cv2.convexHull(approx).reshape(-1, 2)
        # If hull already 4 points, return
        if hull.shape[0] == 4:
            return hull.astype(np.float32)

        # Try to approximate hull down to 4 points by increasing epsilon
        for eps_mul in [0.02, 0.04, 0.06, 0.08, 0.1, 0.2]:
            approx2 = cv2.approxPolyDP(hull.reshape(-1, 1, 2), eps_mul * cv2.arcLength(hull.reshape(-1, 1, 2), True), True)
            if approx2.shape[0] == 4:
                return approx2.reshape(4, 2).astype(np.float32)

    # Fallback to minAreaRect
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype=np.float32)


def detect_board_border(image: np.ndarray, kernel_sizes=(31, 51, 71), min_area_ratio=0.03) -> np.ndarray:
    """Detect the board by finding a large rectangular border via morphology.

    This function attempts to remove small details (chess pieces) by
    morphological closing/opening with progressively larger kernels and
    returns a 4-point quad (float32) if a sufficiently large contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try both normal and inverted binary images (border might be light or dark)
    for invert in (False, True):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            th = 255 - th

        img_area = image.shape[0] * image.shape[1]

        for k in kernel_sizes:
            k = int(k)
            if k <= 1:
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area < min_area_ratio * img_area:
                    continue

                perim = cv2.arcLength(cnt, True)
                eps = 0.02 * perim
                approx = cv2.approxPolyDP(cnt, eps, True)
                if approx.shape[0] == 4:
                    return approx.reshape(4, 2).astype(np.float32)

                # Try convex hull simplification
                hull = cv2.convexHull(cnt).reshape(-1, 2)
                if hull.shape[0] >= 4:
                    for eps_mul in (0.02, 0.04, 0.06, 0.08, 0.12):
                        approx2 = cv2.approxPolyDP(hull.reshape(-1, 1, 2), eps_mul * cv2.arcLength(hull.reshape(-1, 1, 2), True), True)
                        if approx2.shape[0] == 4:
                            return approx2.reshape(4, 2).astype(np.float32)

                # Last resort: axis-aligned min area rect
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                return np.array(box, dtype=np.float32)

    return None


def _fit_line_coeff_from_points(pts: np.ndarray):
    """Fit a line to pts (Nx2) and return coefficients (a,b,c) for ax+by=c."""
    if pts is None or len(pts) < 2:
        return None
    try:
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        # cv2.fitLine returns 1D arrays; extract scalars explicitly to avoid
        # future NumPy deprecation warnings about array-to-scalar conversion.
        vx = float(np.asarray(vx).flatten()[0])
        vy = float(np.asarray(vy).flatten()[0])
        x0 = float(np.asarray(x0).flatten()[0])
        y0 = float(np.asarray(y0).flatten()[0])
        a = -vy
        b = vx
        c = a * x0 + b * y0
        return (a, b, c)
    except Exception:
        return None


def refine_quad_via_edge_fits(quad_pts: np.ndarray, image: np.ndarray, strip_width: int = 32) -> np.ndarray:
    """Refine a 4-point quad by fitting lines to each side using local edge points.

    Returns either a refined (4,2) float32 array or the original quad if no
    robust refinement is available.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return quad_pts

    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    lines_coeff = [None] * 4

    for i in range(4):
        p1 = tuple(np.round(quad_pts[i]).astype(int))
        p2 = tuple(np.round(quad_pts[(i + 1) % 4]).astype(int))
        mask = np.zeros_like(edges)
        cv2.line(mask, p1, p2, 255, thickness=max(3, strip_width))
        masked = cv2.bitwise_and(edges, mask)

        ys, xs = np.nonzero(masked)
        if len(xs) >= 2:
            pts = np.column_stack((xs, ys)).astype(np.float32)
            coeff = _fit_line_coeff_from_points(pts)
            if coeff is not None:
                lines_coeff[i] = coeff
                continue

        # Try a larger strip if nothing found
        mask2 = np.zeros_like(edges)
        cv2.line(mask2, p1, p2, 255, thickness=max(5, strip_width * 2))
        masked2 = cv2.bitwise_and(edges, mask2)
        ys2, xs2 = np.nonzero(masked2)
        if len(xs2) >= 2:
            pts2 = np.column_stack((xs2, ys2)).astype(np.float32)
            coeff = _fit_line_coeff_from_points(pts2)
            if coeff is not None:
                lines_coeff[i] = coeff

    # Compute intersections of adjacent fitted lines to get refined corners
    refined = []
    for i in range(4):
        l_prev = lines_coeff[(i - 1) % 4]
        l_curr = lines_coeff[i]
        if l_prev is None or l_curr is None:
            refined.append(quad_pts[i])
            continue
        a1, b1, c1 = l_prev
        a2, b2, c2 = l_curr
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            refined.append(quad_pts[i])
            continue
        x = (c1 * b2 - c2 * b1) / denom
        y = (a1 * c2 - a2 * c1) / denom
        # Clamp to image bounds
        x = max(0.0, min(float(w - 1), float(x)))
        y = max(0.0, min(float(h - 1), float(y)))
        refined.append([x, y])

    refined = np.array(refined, dtype=np.float32)

    # Final subpixel refinement if possible
    try:
        corners = refined.reshape(-1, 1, 2).astype(np.float32)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        refined = corners.reshape(-1, 2).astype(np.float32)
    except Exception:
        pass

    return refined


def _bilinear_sample(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    dx = x - x0; dy = y - y0
    I00 = float(img[y0, x0]); I10 = float(img[y0, x0 + 1]); I01 = float(img[y0 + 1, x0]); I11 = float(img[y0 + 1, x0 + 1])
    return (I00 * (1 - dx) * (1 - dy) + I10 * dx * (1 - dy) + I01 * (1 - dx) * dy + I11 * dx * dy)


def refine_quad_via_gradient_search(quad_pts: np.ndarray, image: np.ndarray, strip: int = 48, num_samples: int = 36, steps: int = 25) -> np.ndarray:
    """Refine a quad by searching for strong gradient peaks along normals to each side.

    For each side we sample several points along the side, then along a normal
    search for the distance with maximum gradient magnitude. We fit a line to
    these peak locations and intersect adjacent lines to obtain refined corners.
    Returns refined (4,2) array or None on failure.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None

    h, w = gray.shape[:2]
    # Precompute gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    center = quad_pts.mean(axis=0)
    line_coeffs = []

    for i in range(4):
        p1 = quad_pts[i]
        p2 = quad_pts[(i + 1) % 4]
        s = p2 - p1
        L = np.linalg.norm(s)
        if L < 1.0:
            return None
        t = s / L
        # normal direction (unit)
        n = np.array([-t[1], t[0]], dtype=np.float32)
        # ensure normal points inward toward center
        m = 0.5 * (p1 + p2)
        if np.dot(center - m, n) < 0:
            n = -n

        peak_points = []

        for si in range(num_samples):
            frac = (si + 0.5) / float(num_samples)
            base = p1 + frac * s
            best_val = -1.0
            best_pt = None
            # search along normal from -strip..+strip
            for k in range(steps):
                d = -strip + (2.0 * strip) * (k / float(steps - 1))
                x = float(base[0] + d * n[0])
                y = float(base[1] + d * n[1])
                val = _bilinear_sample(grad, x, y)
                if val > best_val:
                    best_val = val
                    best_pt = (x, y)

            if best_pt is not None:
                peak_points.append(best_pt)

        if len(peak_points) < max(4, num_samples // 4):
            # Not enough peaks found
            return None

        pts = np.array(peak_points, dtype=np.float32)
        # fit line using cv2.fitLine
        try:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(np.asarray(vx).flatten()[0])
            vy = float(np.asarray(vy).flatten()[0])
            x0 = float(np.asarray(x0).flatten()[0])
            y0 = float(np.asarray(y0).flatten()[0])
            a = -vy; b = vx; c = a * x0 + b * y0
            line_coeffs.append((a, b, c))
        except Exception:
            return None

    if len(line_coeffs) != 4:
        return None

    # intersect adjacent lines to get corners
    refined = []
    for i in range(4):
        a1, b1, c1 = line_coeffs[i]
        a2, b2, c2 = line_coeffs[(i + 1) % 4]
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            return None
        x = (c1 * b2 - c2 * b1) / denom
        y = (a1 * c2 - a2 * c1) / denom
        x = max(0.0, min(float(w - 1), float(x)))
        y = max(0.0, min(float(h - 1), float(y)))
        refined.append([x, y])

    return np.array(refined, dtype=np.float32)


def inset_quad(quad_pts: np.ndarray, inset_frac: float = 0.06) -> np.ndarray:
    """Produce an inset (shrunken) quad by offsetting each side inward.

    Insets each side by `inset_frac * min_side_length` and returns the
    intersection of the offset lines as a new quad. Returns the original quad
    on failure.
    """
    try:
        quad = np.array(quad_pts, dtype=np.float32).reshape(4, 2)
    except Exception:
        return quad_pts

    # compute side lengths
    lens = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    min_side = max(1.0, min(lens))
    inset_dist = float(inset_frac) * min_side

    center = quad.mean(axis=0)
    lines = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        s = p2 - p1
        L = np.linalg.norm(s)
        if L < 1e-6:
            return quad_pts
        t = s / L
        n = np.array([-t[1], t[0]], dtype=np.float32)
        mid = 0.5 * (p1 + p2)
        # ensure normal points inward
        if np.dot(center - mid, n) < 0:
            n = -n
        a = float(n[0]); b = float(n[1]); c = float(np.dot(n, mid))
        c_in = c + inset_dist
        lines.append((a, b, c_in))

    refined = []
    for i in range(4):
        a1, b1, c1 = lines[i]
        a2, b2, c2 = lines[(i + 1) % 4]
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            return quad_pts
        x = (c1 * b2 - c2 * b1) / denom
        y = (a1 * c2 - a2 * c1) / denom
        refined.append([max(0.0, x), max(0.0, y)])

    return np.array(refined, dtype=np.float32)


def find_outer_from_inner_corners(corners: np.ndarray, pattern=(7, 7)) -> np.ndarray:
    """Given inner chessboard corners (N,1,2) or (N,2), compute outer 4 corner points.

    Returns a (4,2) float32 array of outer corners in arbitrary order.
    """
    pts = corners.reshape(-1, 2).astype(np.float32)
    rows, cols = pattern
    if pts.shape[0] != rows * cols:
        raise ValueError("Unexpected number of inner corners")

    grid = pts.reshape(rows, cols, 2)

    # Compute average step vectors along rows and columns
    row_steps = grid[:, 1:, :] - grid[:, :-1, :]
    col_steps = grid[1:, :, :] - grid[:-1, :, :]
    row_vec = row_steps.mean(axis=(0, 1))
    col_vec = col_steps.mean(axis=(0, 1))

    tl_inner = grid[0, 0]
    tr_inner = grid[0, -1]
    bl_inner = grid[-1, 0]
    br_inner = grid[-1, -1]

    # Extrapolate outer corners by half a step in both directions
    tl_outer = tl_inner - 0.5 * row_vec - 0.5 * col_vec
    tr_outer = tr_inner + 0.5 * row_vec - 0.5 * col_vec
    bl_outer = bl_inner - 0.5 * row_vec + 0.5 * col_vec
    br_outer = br_inner + 0.5 * row_vec + 0.5 * col_vec

    return np.array([tl_outer, tr_outer, br_outer, bl_outer], dtype=np.float32)


def detect_board_via_chessboard(image: np.ndarray, pattern=(7, 7)) -> np.ndarray:
    """Try to detect the board using OpenCV's findChessboardCorners.

    Returns 4 outer corner points (float32) or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not found:
        return None

    # Refine corner locations
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    outer = find_outer_from_inner_corners(corners, pattern=pattern)
    return outer


def intersect_rho_theta(l1, l2):
    """Intersect two lines in (rho,theta) Hough form. Returns (x,y)."""
    rho1, th1 = l1
    rho2, th2 = l2
    A = np.array([[np.cos(th1), np.sin(th1)], [np.cos(th2), np.sin(th2)]])
    b = np.array([rho1, rho2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x, y = np.linalg.solve(A, b)
    return np.array([x, y], dtype=np.float32)


def detect_board_via_hough(image: np.ndarray) -> np.ndarray:
    """Detect 4 board corners using clustered Hough lines (robust 1D theta clustering).

    Returns a (4,2) float32 array (unordered) or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    edges = cv2.Canny(blur, max(1, 0.66 * v), 1.33 * v)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lines is None:
        return None

    lines = lines.reshape(-1, 2)
    thetas = lines[:, 1]

    # 1D k-means on theta to find two dominant orientations
    c1, c2 = float(thetas.min()), float(thetas.max())
    for _ in range(20):
        d1 = np.abs(thetas - c1)
        d2 = np.abs(thetas - c2)
        labels = (d2 < d1).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            break
        nc1 = thetas[labels == 0].mean() if np.any(labels == 0) else c1
        nc2 = thetas[labels == 1].mean() if np.any(labels == 1) else c2
        if abs(nc1 - c1) < 1e-6 and abs(nc2 - c2) < 1e-6:
            break
        c1, c2 = nc1, nc2

    # Group lines by label
    group0 = lines[labels == 0]
    group1 = lines[labels == 1]

    if len(group0) < 2 or len(group1) < 2:
        return None

    # Pick the two most extreme rhos in each group (min and max)
    def extremes(group):
        idx = np.argsort(group[:, 0])
        return group[idx[0]], group[idx[-1]]

    g0_min, g0_max = extremes(group0)
    g1_min, g1_max = extremes(group1)

    # Compute the four intersections
    pairs = [(g0_min, g1_min), (g0_min, g1_max), (g0_max, g1_min), (g0_max, g1_max)]
    pts = []
    for a, b in pairs:
        p = intersect_rho_theta(a, b)
        if p is None:
            return None
        pts.append(p)

    pts = np.array(pts, dtype=np.float32)

    # Sanity-check: area should be a reasonable fraction of image area
    area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
    img_area = image.shape[0] * image.shape[1]
    if area < 0.005 * img_area:  # too small
        return None

    return pts


def detect_board_via_hough_segments(image: np.ndarray) -> np.ndarray:
    """Detect board quad by using Hough probabilistic segments + orientation clustering.

    Returns a (4,2) float32 array or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=20)
    if segs is None or len(segs) < 6:
        return None

    segs = segs.reshape(-1, 4)
    mids = np.column_stack(((segs[:, 0] + segs[:, 2]) * 0.5, (segs[:, 1] + segs[:, 3]) * 0.5))
    centroid = mids.mean(axis=0)
    dists = np.linalg.norm(mids - centroid, axis=1)
    med = float(np.median(dists))
    thr = max(med * 1.6, min(image.shape[:2]) * 0.15)
    good_idx = np.where(dists <= thr)[0]
    if len(good_idx) < 6:
        good_idx = np.arange(len(segs))

    segs = segs[good_idx]

    angles = np.arctan2((segs[:, 3] - segs[:, 1]).astype(np.float32), (segs[:, 2] - segs[:, 0]).astype(np.float32))
    thetas = np.mod(angles, np.pi)

    # simple 1D k-means into 2 clusters
    c1, c2 = float(thetas.min()), float(thetas.max())
    for _ in range(30):
        d1 = np.abs(thetas - c1)
        d2 = np.abs(thetas - c2)
        labels = (d2 < d1).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            break
        nc1 = thetas[labels == 0].mean() if np.any(labels == 0) else c1
        nc2 = thetas[labels == 1].mean() if np.any(labels == 1) else c2
        if abs(nc1 - c1) < 1e-6 and abs(nc2 - c2) < 1e-6:
            break
        c1, c2 = nc1, nc2

    group0 = segs[labels == 0]
    group1 = segs[labels == 1]
    if len(group0) < 2 or len(group1) < 2:
        return None

    def fit_line_from_segments(group):
        # compute midpoint projection along normal for each segment
        mids_g = np.column_stack(((group[:, 0] + group[:, 2]) * 0.5, (group[:, 1] + group[:, 3]) * 0.5))
        ps = []
        for s in group:
            x1, y1, x2, y2 = s
            t = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            L = np.linalg.norm(t)
            if L < 1e-6:
                continue
            n = np.array([-t[1], t[0]], dtype=np.float32) / L
            mid = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            p = float(np.dot(n, mid))
            ps.append((p, s))
        if not ps:
            return None, None
        ps = sorted(ps, key=lambda x: x[0])
        # take extreme small group and extreme large group combine into two line fits
        n_take = max(2, len(ps) // 6)
        low_segs = np.vstack([item[1] for item in ps[:n_take]])
        high_segs = np.vstack([item[1] for item in ps[-n_take:]])

        def fit_from_segs(segs_block):
            pts = []
            for s in segs_block:
                pts.append([s[0], s[1]])
                pts.append([s[2], s[3]])
            pts = np.array(pts, dtype=np.float32)
            coeff = _fit_line_coeff_from_points(pts)
            return coeff

        low_line = fit_from_segs(low_segs)
        high_line = fit_from_segs(high_segs)
        return low_line, high_line

    g0_min, g0_max = fit_line_from_segments(group0)
    g1_min, g1_max = fit_line_from_segments(group1)

    if g0_min is None or g0_max is None or g1_min is None or g1_max is None:
        return None

    # intersections
    def intersect_ab(a1, b1, c1, a2, b2, c2):
        D = a1 * b2 - a2 * b1
        if abs(D) < 1e-6:
            return None
        x = (c1 * b2 - c2 * b1) / D
        y = (a1 * c2 - a2 * c1) / D
        return np.array([x, y], dtype=np.float32)

    pts = []
    pairs = [(g0_min, g1_min), (g0_min, g1_max), (g0_max, g1_min), (g0_max, g1_max)]
    for a, b in pairs:
        if a is None or b is None:
            return None
        p = intersect_ab(a[0], a[1], a[2], b[0], b[1], b[2])
        if p is None:
            return None
        pts.append(p)

    pts = np.array(pts, dtype=np.float32)
    area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
    img_area = float(image.shape[0] * image.shape[1])
    if area < 0.002 * img_area:
        return None

    return pts


def process_image_file(in_path: Path, out_dir: Path) -> tuple[bool, str]:
    """Process a single image file: detect board, warp, draw grid, and save.

    Returns (success, error_message).
    """
    try:
        image = load_image(str(in_path))

        # Multi-scale detector helper: try detector at multiple scales and map
        # detected coordinates back to the original image space.
        def run_detector_multiscale(detector_fn, img, scales=(1.0, 0.8, 0.6, 1.2)):
            h0, w0 = img.shape[:2]
            for s in scales:
                if s == 1.0:
                    img_s = img
                else:
                    nw = max(8, int(round(w0 * s)))
                    nh = max(8, int(round(h0 * s)))
                    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                    img_s = cv2.resize(img, (nw, nh), interpolation=interp)
                try:
                    pts = detector_fn(img_s)
                except Exception:
                    pts = None
                if pts is not None:
                    try:
                        pts = np.asarray(pts, dtype=np.float32)
                        # Map back to original image coords
                        if s != 1.0:
                            pts = pts.astype(np.float32) / float(s)
                        return pts
                    except Exception:
                        return None
            return None

        # Try chessboard-corner based detection first (works when inner corners are visible)
        warped = None
        last_src_for_affine = None
        last_reason_for_affine = ""
        debug_dir = Path(CONFIG["output_dir"]) / "debug" / in_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Try multiple detectors and score candidate warps, then pick the best
        best_score = -1.0
        best_warp = None
        best_method = None
        best_src = None

        w, h = CONFIG["output_size"]
        min_warp_score = CONFIG.get("min_warp_score", 0.12)

        # 1) Chessboard-corners detector (multi-scale)
        src_pts = run_detector_multiscale(detect_board_via_chessboard, image)
        if src_pts is not None:
            src_ordered = order_points(src_pts)
            # refine detected chessboard outer corners (helps for imperfect photos)
            try:
                refined = refine_quad_via_edge_fits(src_ordered, image, strip_width=24)
                if refined is not None and refined.shape == (4, 2):
                    src_ordered = order_points(refined)
            except Exception:
                pass
            dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            M, reason = compute_and_validate_homography(src_ordered, dst_pts, (w, h), image.shape)
            overlay = image.copy()
            for p in src_ordered.astype(int):
                cv2.circle(overlay, tuple(p), 8, (0, 255, 0), -1)
            cv2.imwrite(str(debug_dir / f"{in_path.stem}_chess_corners.jpg"), overlay)
            if M is None:
                print(f"Chessboard homography invalid: {reason}")
                last_src_for_affine = src_ordered
                last_reason_for_affine = reason
            else:
                candidate = cv2.warpPerspective(image, M, (w, h))
                score = score_warped_image(candidate)
                cv2.imwrite(str(debug_dir / f"{in_path.stem}_chess_candidate.jpg"), candidate)
                if score > best_score:
                    best_score = score
                    best_warp = candidate
                    best_method = "chessboard"
                    best_src = src_ordered

        # 1.5) Border-based detection: try to find a large decorative border
        # or outer box around the board using morphology (helps when pieces
        # occlude inner corners).
        try:
            src_pts = run_detector_multiscale(detect_board_border, image)
            if src_pts is not None:
                src_ordered = order_points(src_pts)
                dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

                # Save overlay for the original border detection
                overlay = image.copy()
                cv2.polylines(overlay, [src_ordered.astype(np.int32).reshape(-1, 1, 2)], True, (0, 128, 255), 4)
                cv2.imwrite(str(debug_dir / f"{in_path.stem}_border.jpg"), overlay)

                # Evaluate the original border candidate
                M_orig, reason_orig = compute_and_validate_homography(src_ordered, dst_pts, (w, h), image.shape)
                score_orig = -1.0
                if M_orig is not None:
                    cand_orig = cv2.warpPerspective(image, M_orig, (w, h))
                    score_orig = score_warped_image(cand_orig)
                    cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_candidate.jpg"), cand_orig)

                # Try to find a tighter inner border by examining the warped candidate
                score_inner = -1.0
                cand_inner = None
                try:
                    inner_quad = detect_board_quad(cand_orig)
                    if inner_quad is not None:
                        # Map inner-quad (in warped coords) back to original image
                        try:
                            Minv = np.linalg.inv(M_orig)
                            src_from_inner = cv2.perspectiveTransform(inner_quad.reshape(-1, 1, 2).astype(np.float32), Minv.astype(np.float32)).reshape(-1, 2)
                            src_from_inner = order_points(src_from_inner)
                            overlay3 = image.copy()
                            cv2.polylines(overlay3, [src_from_inner.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 200), 4)
                            cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_innerwarp.jpg"), overlay3)
                            M_inner, reason_inner = compute_and_validate_homography(src_from_inner, dst_pts, (w, h), image.shape)
                            if M_inner is not None:
                                cand_inner = cv2.warpPerspective(image, M_inner, (w, h))
                                score_inner = score_warped_image(cand_inner)
                                cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_candidate_inner.jpg"), cand_inner)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Try a stronger gradient-based refinement of the detected border
                try:
                    refined_grad = refine_quad_via_gradient_search(src_ordered, image, strip=48, num_samples=36, steps=25)
                except Exception:
                    refined_grad = None

                score_refined = -1.0
                M_refined = None
                cand_ref = None
                if refined_grad is not None:
                    src_ref = order_points(refined_grad)
                    overlay2 = image.copy()
                    cv2.polylines(overlay2, [src_ref.astype(np.int32).reshape(-1, 1, 2)], True, (255, 128, 0), 4)
                    cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_refined.jpg"), overlay2)
                    M_refined, reason_refined = compute_and_validate_homography(src_ref, dst_pts, (w, h), image.shape)
                    if M_refined is not None:
                        cand_ref = cv2.warpPerspective(image, M_refined, (w, h))
                        score_refined = score_warped_image(cand_ref)
                        cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_candidate_refined.jpg"), cand_ref)

                # Try an inset (shrunk) quad based on the detected border
                score_inset = -1.0
                cand_inset = None
                try:
                    inset_q = inset_quad(src_ordered, inset_frac=0.06)
                    if inset_q is not None:
                        overlay4 = image.copy()
                        cv2.polylines(overlay4, [order_points(inset_q).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 4)
                        cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_inset.jpg"), overlay4)
                        M_inset, reason_inset = compute_and_validate_homography(order_points(inset_q), dst_pts, (w, h), image.shape)
                        if M_inset is not None:
                            cand_inset = cv2.warpPerspective(image, M_inset, (w, h))
                            score_inset = score_warped_image(cand_inset)
                            cv2.imwrite(str(debug_dir / f"{in_path.stem}_border_candidate_inset.jpg"), cand_inset)
                except Exception:
                    score_inset = -1.0

                # Choose the best among original, inner-warp-derived, gradient-refined, and inset
                best_local_score = score_orig
                best_local_warp = cand_orig
                best_local_src = src_ordered
                best_local_method = "border"

                if score_inner > best_local_score:
                    best_local_score = score_inner
                    best_local_warp = cand_inner
                    best_local_src = src_from_inner if 'src_from_inner' in locals() else best_local_src
                    best_local_method = "border_inner"

                if score_refined > best_local_score:
                    best_local_score = score_refined
                    best_local_warp = cand_ref
                    best_local_src = src_ref if 'src_ref' in locals() else best_local_src
                    best_local_method = "border_refined"

                if score_inset > best_local_score:
                    best_local_score = score_inset
                    best_local_warp = cand_inset
                    best_local_src = order_points(inset_q) if 'inset_q' in locals() else best_local_src
                    best_local_method = "border_inset"

                if best_local_score > best_score:
                    best_score = best_local_score
                    best_warp = best_local_warp
                    best_method = best_local_method
                    best_src = best_local_src

                # If neither produced a valid homography, record last source for affine
                if (M_orig is None or score_orig < 0.0) and (M_refined is None or score_refined < 0.0):
                    last_src_for_affine = src_ordered
                    last_reason_for_affine = reason_orig if M_orig is None else (reason_refined if M_refined is None else "")
        except Exception as e:
            print(f"Border detection failed: {e}")

        # 2) Square-based hull heuristic
        binary_image = apply_threshold(image)
        edges_img = detect_edges(binary_image)
        lines = detect_lines(edges_img, CONFIG["hough_params"])
        contours = find_contours(lines, binary_image.shape)
        valid_squares = filter_squares(contours, CONFIG["square_area_range"], CONFIG["square_length_threshold"])

        MIN_SQUARES = CONFIG.get("min_squares_for_board", 20)
        MIN_HULL_RATIO = CONFIG.get("min_hull_area_ratio", 0.03)

        if valid_squares and len(valid_squares) >= MIN_SQUARES:
            pts_all = np.array([pt for sq in valid_squares for pt in sq], dtype=np.float32).reshape(-1, 2)
            hull = cv2.convexHull(pts_all)
            hull_area = cv2.contourArea(hull)
            img_area = image.shape[0] * image.shape[1]
            hull_ratio = hull_area / img_area if img_area > 0 else 0
            overlay = image.copy()
            cv2.polylines(overlay, [hull.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 4)
            cv2.imwrite(str(debug_dir / f"{in_path.stem}_hull.jpg"), overlay)
            if hull_ratio >= MIN_HULL_RATIO:
                try:
                    candidate, _ = perspective_correction(image, valid_squares, CONFIG["output_size"])
                    score = score_warped_image(candidate)
                    cv2.imwrite(str(debug_dir / f"{in_path.stem}_square_candidate.jpg"), candidate)
                    if score > best_score:
                        best_score = score
                        best_warp = candidate
                        best_method = "square_hull"
                        # approximate src from hull
                        try:
                            approx2 = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.02 * cv2.arcLength(hull.reshape(-1, 1, 2), True), True)
                            if approx2.shape[0] == 4:
                                best_src = order_points(approx2.reshape(4, 2).astype(np.float32))
                                # refine hull approximation
                                try:
                                    refined = refine_quad_via_edge_fits(best_src, image, strip_width=28)
                                    if refined is not None and refined.shape == (4, 2):
                                        best_src = order_points(refined)
                                except Exception:
                                    pass
                        except Exception:
                            best_src = None
                except Exception as e:
                    print(f"Square-based perspective failed: {e}")
                    try:
                        pts_all = np.array([pt for sq in valid_squares for pt in sq], dtype=np.float32).reshape(-1, 2)
                        hull = cv2.convexHull(pts_all).reshape(-1, 2)
                        if hull.shape[0] >= 4:
                            approx2 = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.02 * cv2.arcLength(hull.reshape(-1, 1, 2), True), True)
                            if approx2.shape[0] == 4:
                                last_src_for_affine = order_points(approx2.reshape(4, 2).astype(np.float32))
                    except Exception:
                        pass

        # 3) Hough-lines based detector (try Hough segments first), multi-scale
        src_pts = run_detector_multiscale(lambda im: detect_board_via_hough_segments(im) or detect_board_via_hough(im), image)
        if src_pts is not None:
            src_ordered = order_points(src_pts)
            try:
                refined = refine_quad_via_edge_fits(src_ordered, image, strip_width=28)
                if refined is not None and refined.shape == (4, 2):
                    src_ordered = order_points(refined)
            except Exception:
                pass
            dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            M, reason = compute_and_validate_homography(src_ordered, dst_pts, (w, h), image.shape)
            overlay = image.copy()
            for p in src_ordered.astype(int):
                cv2.circle(overlay, tuple(p), 8, (255, 0, 0), -1)
            cv2.imwrite(str(debug_dir / f"{in_path.stem}_hough.jpg"), overlay)
            if M is None:
                print(f"Hough homography invalid: {reason}")
                last_src_for_affine = src_ordered
                last_reason_for_affine = reason
            else:
                candidate = cv2.warpPerspective(image, M, (w, h))
                score = score_warped_image(candidate)
                cv2.imwrite(str(debug_dir / f"{in_path.stem}_hough_candidate.jpg"), candidate)
                if score > best_score:
                    best_score = score
                    best_warp = candidate
                    best_method = "hough"
                    best_src = src_ordered

        # 4) Morphological quad detector (final fallback candidate), multi-scale
        src_pts = run_detector_multiscale(detect_board_quad, image)
        if src_pts is not None:
            src_ordered = order_points(src_pts)
            try:
                refined = refine_quad_via_edge_fits(src_ordered, image, strip_width=36)
                if refined is not None and refined.shape == (4, 2):
                    src_ordered = order_points(refined)
            except Exception:
                pass
            dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            M, reason = compute_and_validate_homography(src_ordered, dst_pts, (w, h), image.shape)
            if M is None:
                print(f"Final fallback homography invalid: {reason}")
                last_src_for_affine = src_ordered
                last_reason_for_affine = reason
            else:
                candidate = cv2.warpPerspective(image, M, (w, h))
                score = score_warped_image(candidate)
                cv2.imwrite(str(debug_dir / f"{in_path.stem}_morph_candidate.jpg"), candidate)
                if score > best_score:
                    best_score = score
                    best_warp = candidate
                    best_method = "morph"
                    best_src = src_ordered

        # Choose best candidate or try affine fallback
        if best_warp is not None and best_score >= min_warp_score:
            warped = best_warp
            print(f"Selected {best_method} candidate (score={best_score:.3f})")
        else:
            # Affine fallback attempt
            if last_src_for_affine is not None:
                try:
                    src_tri = np.array([last_src_for_affine[0], last_src_for_affine[1], last_src_for_affine[3]], dtype=np.float32)
                    dst_tri = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)
                    A = cv2.getAffineTransform(src_tri, dst_tri)
                    candidate = cv2.warpAffine(image, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    cv2.imwrite(str(debug_dir / f"{in_path.stem}_affine_fallback.jpg"), candidate)
                    score = score_warped_image(candidate)
                    if score >= min_warp_score:
                        warped = candidate
                        print(f"Affine fallback accepted (score={score:.3f})")
                    else:
                        raise ValueError(f"No valid warp candidate found; best_score={best_score:.3f}, affine_score={score:.3f}")
                except Exception as e:
                    raise
            else:
                raise ValueError(f"No valid warp candidate found; best_score={best_score:.3f}")

        # If all perspective methods failed but we have a last source quad, try affine fallback
        if warped is None and last_src_for_affine is not None:
            try:
                w, h = CONFIG["output_size"]
                # use TL, TR, BL -> map to (0,0),(w,0),(0,h)
                src_tri = np.array([last_src_for_affine[0], last_src_for_affine[1], last_src_for_affine[3]], dtype=np.float32)
                dst_tri = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)
                A = cv2.getAffineTransform(src_tri, dst_tri)
                warped = cv2.warpAffine(image, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # save debug overlay
                overlay = image.copy()
                for p in last_src_for_affine.astype(int):
                    cv2.circle(overlay, tuple(p), 6, (0, 0, 255), -1)
                cv2.imwrite(str(debug_dir / f"{in_path.stem}_affine_fallback.jpg"), overlay)
            except Exception as e:
                print(f"Affine fallback failed: {e}")

        # Draw grid and save
        final = draw_chessboard_grid(warped, CONFIG["board_size"]) if warped is not None else image
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / in_path.name
        cv2.imwrite(str(out_path), final)
        print(f"Saved: {out_path}")
        return True, ""

    except Exception as e:
        print(f"Failed processing {in_path.name}: {e}")
        return False, str(e)

def main():
    input_dir = Path(CONFIG["input_dir"])
    output_dir = Path(CONFIG["output_dir"])

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    files = [p for p in input_dir.iterdir() if p.suffix.lower() in CONFIG["file_extensions"]]
    if not files:
        print(f"No image files found in {input_dir}")
        return

    success = 0
    failures = []

    for p in files:
        print(f"Processing: {p.name}")
        ok, err = process_image_file(p, output_dir)
        if ok:
            success += 1
        else:
            failures.append((p.name, err))

    print(f"Processed {len(files)} files: {success} succeeded, {len(failures)} failed")
    if failures:
        for name, err in failures:
            print(f" - {name}: {err}")

if __name__ == "__main__":
    main()
