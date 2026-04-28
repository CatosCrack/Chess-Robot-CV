"""
perspective_correction.py  –  robust chessboard boundary + grid overlay

Validated against:
  • Strongly angled shots on a plain table (perspective distortion)
  • Chess box with thick decorative wooden frame (inset playing grid)
  • Dark background with metallic pieces (low contrast squares)
  • Near-overhead shot, two-tone wood board, no pieces

Pipeline
────────
Stage 1 – Contour detection
    Build three edge maps (Otsu+Canny, Adaptive+Canny, Direct Canny) and find
    the best-scoring quadrilateral across all of them.  Scoring weights area,
    squareness, and corner-angle closeness to 90°.

Stage 2 – Hough refinement (always attempted)
    Warp the stage-1 quad to a square, then run multi-threshold Hough line
    detection to find the actual playing grid boundary inside that warp.
    The outermost detected H/V line pair gives the true grid corners, which
    are mapped back to original-image space via the inverse transform.

Stage 3 – Direct Hough fallback
    If stage 1 finds nothing, run Hough directly on the original image and
    intersect the outermost H/V lines to produce 4 corners.

Corner ordering
    Uses angle-from-centroid, which is robust to the 3-above/1-below split
    failure that hit previous implementations on angled boards.

Usage
─────
    python perspective_correction.py <image_path>
    python perspective_correction.py <image_path> --debug
    python perspective_correction.py              # uses ./test_images/board1.jpg
"""

from __future__ import annotations
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ── Output size of the warped board image (pixels × pixels) ──────────────────
WARP_SIZE = 1200


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _load(path: str) -> np.ndarray:
    """Load image as BGR.  Falls back to Pillow for webp and other formats."""
    bgr = cv2.imread(str(path))
    if bgr is not None:
        return bgr
    if not _PIL_AVAILABLE:
        raise FileNotFoundError(f"cv2 could not read '{path}' and Pillow is not installed.")
    img = _PILImage.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Return four points ordered (TL, TR, BR, BL) using angle-from-centroid.

    Robust to perspective cases where points don't split cleanly into
    "above/below centroid" (the common failure of sum/diff methods on
    angled boards).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    cx, cy = pts.mean(axis=0)
    angles = (np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) + 360) % 360
    idx = np.argsort(angles)          # → [BR, BL, TL, TR]
    o = pts[idx]
    return np.array([o[2], o[3], o[0], o[1]], dtype=np.float32)  # TL TR BR BL


def _cluster(coords: list[float], gap: float) -> list[float]:
    """Merge coords within `gap` of each other; return cluster means."""
    if not coords:
        return []
    coords = sorted(coords)
    clusters: list[float] = []
    cur = [coords[0]]
    for c in coords[1:]:
        if c - cur[-1] < gap:
            cur.append(c)
        else:
            clusters.append(float(np.mean(cur)))
            cur = [c]
    clusters.append(float(np.mean(cur)))
    return clusters


def _line_intersection(
    l1: tuple[float, float, float, float],
    l2: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Intersection of two infinite lines, each given as (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _extend_segment(
    seg: tuple[float, float, float, float, float, float],
    length: float = 10_000.0,
) -> tuple[float, float, float, float]:
    """Extend a line segment (x1,y1,x2,y2,…) to a long line through the same points."""
    x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx * dx + dy * dy) + 1e-9
    dx, dy = dx / d, dy / d
    return x1 - dx * length, y1 - dy * length, x1 + dx * length, y1 + dy * length


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 – contour-based quad detection
# ══════════════════════════════════════════════════════════════════════════════

def _build_edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    h, w = gray.shape
    k  = max(3, (int(min(h, w) * 0.006) // 2) * 2 + 1)   # blur kernel, odd ≥ 3
    dk = max(3, (int(min(h, w) * 0.008) // 2) * 2 + 1)   # dilation kernel

    blur   = cv2.GaussianBlur(gray, (k, k), 0)
    kernel = np.ones((dk, dk), np.uint8)

    # Use robust thresholds derived from the blurred gray image (not binary medians)
    med = float(np.median(blur))
    low = max(1.0, 0.66 * med)
    high = min(254.0, 1.33 * med)

    # A: Otsu threshold → Canny (use blur-derived thresholds)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    map_a = cv2.dilate(cv2.Canny(otsu, low, high), kernel, iterations=1)

    # B: Adaptive threshold → Canny (use same blur-derived thresholds)
    block = max(11, (int(min(h, w) * 0.03) // 2) * 2 + 1)
    adap  = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 4,
    )
    map_b = cv2.dilate(cv2.Canny(adap, low, high), kernel, iterations=1)

    # C: Direct Canny on blurred gray (good for strongly angled shots)
    map_c = cv2.dilate(cv2.Canny(blur, low, high), kernel, iterations=2)

    # Suppress small connected components (likely chess pieces) from edge maps.
    # Build a combined binary from Otsu+Adaptive threshold and remove small
    # blobs whose area is much smaller than a board square.
    try:
        combined = cv2.bitwise_or(otsu, adap)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = float(h * w)
        # heuristic: a piece is much smaller than a square (~1/64 of image)
        piece_area_thresh = max(200.0, img_area / 400.0)
        small_mask = np.zeros_like(gray, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0 and area < piece_area_thresh:
                cv2.drawContours(small_mask, [cnt], -1, 255, thickness=-1)
        if small_mask.sum() > 0:
            # dilate mask slightly to cover piece edges
            dk2 = max(3, (int(min(h, w) * 0.01) // 2) * 2 + 1)
            small_mask = cv2.dilate(small_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk2, dk2)), iterations=1)
            map_a[small_mask == 255] = 0
            map_b[small_mask == 255] = 0
            map_c[small_mask == 255] = 0
    except Exception:
        # non-fatal; if piece suppression fails, proceed with original maps
        pass

    return [map_a, map_b, map_c]


def _score_quad(approx: np.ndarray, image_area: float) -> float:
    """
    Score a quadrilateral on three criteria:
      area_frac  – fraction of image covered (0.03–0.99)
      squareness – ratio of opposite-side averages  (sq^1, not sq^2, to allow
                   rectangular chess boxes)
      angle_score – closeness of each corner to 90°

    Returns 0.0 on failure.
    """
    area = cv2.contourArea(approx)
    frac = area / image_area
    if frac < 0.03 or frac > 0.99:
        return 0.0

    pts   = approx.reshape(4, 2).astype(np.float32)
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    if min(sides) < 1.0:
        return 0.0

    pa = (sides[0] + sides[2]) / 2.0
    pb = (sides[1] + sides[3]) / 2.0
    sq = min(pa, pb) / max(pa, pb)

    ang = 0.0
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        deg   = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        ang  += max(0.0, 1.0 - abs(deg - 90.0) / 45.0)
    ang /= 4.0

    return frac * sq * (ang ** 2)


def _find_best_contour_quad(gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Search three edge maps for the highest-scoring quadrilateral.
    Uses RETR_LIST so interior contours (the board inside a scene) are included.
    """
    h, w   = gray.shape
    ia     = float(h * w)
    emaps  = _build_edge_maps(gray)

    best_sc: float          = 0.0
    best_approx: np.ndarray | None = None

    for em in emaps:
        contours, _ = cv2.findContours(em, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < ia * 0.03:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    sc = _score_quad(approx, ia)
                    if sc > best_sc:
                        best_sc     = sc
                        best_approx = approx
                    break   # found a 4-sided fit; no need to try looser epsilon

    return best_approx, best_sc


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – Hough line refinement
# ══════════════════════════════════════════════════════════════════════════════

def _hough_grid_lines(
    img_gray: np.ndarray,
) -> tuple[list, list, list[float], list[float]]:
    """
    Detect chess grid lines using three Canny thresholds and HoughLinesP.
    Returns (all_h_segs, all_v_segs, h_cluster_positions, v_cluster_positions).
    Each segment is (x1, y1, x2, y2, length, midpoint).
    """
    blur  = cv2.GaussianBlur(img_gray, (5, 5), 0)
    short = min(img_gray.shape)

    all_h: list = []
    all_v: list = []

    for lo, hi, thr_frac, minL_frac, maxG_frac in [
        (15, 60,  0.25, 0.30, 0.06),
        (10, 40,  0.15, 0.18, 0.10),
        (5,  25,  0.08, 0.10, 0.12),
    ]:
        edges = cv2.Canny(blur, lo, hi)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=int(short * thr_frac),
            minLineLength=int(short * minL_frac),
            maxLineGap=int(short * maxG_frac),
        )
        if lines is None:
            continue
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle  = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            if angle < 20:
                all_h.append((x1, y1, x2, y2, length, (y1 + y2) / 2.0))
            elif angle > 70:
                all_v.append((x1, y1, x2, y2, length, (x1 + x2) / 2.0))

    gap   = short / 18.0
    h_pos = _cluster([s[5] for s in all_h], gap)
    v_pos = _cluster([s[5] for s in all_v], gap)

    return all_h, all_v, h_pos, v_pos


def _corners_from_hough(
    all_h: list,
    all_v: list,
    h_pos: list[float],
    v_pos: list[float],
    size: float,
) -> np.ndarray | None:
    """
    Given clustered Hough line positions, compute the 4 corners of the
    playing grid by intersecting the outermost H/V line pairs.
    Falls back to axis-aligned corners if line intersection fails.
    Returns shape (4, 2) float32 in [TL, TR, BR, BL] order, or None.
    """
    if len(h_pos) < 2 or len(v_pos) < 2:
        return None

    span_h = max(h_pos) - min(h_pos)
    span_v = max(v_pos) - min(v_pos)
    if span_h < size * 0.20 or span_v < size * 0.20:
        return None

    top_y  = min(h_pos);  bot_y  = max(h_pos)
    left_x = min(v_pos);  right_x = max(v_pos)
    tol    = size / 15.0

    def _best_seg(segs: list, target: float, is_h: bool):
        key  = (lambda s: (s[1] + s[3]) / 2.0) if is_h else (lambda s: (s[0] + s[2]) / 2.0)
        near = [s for s in segs if abs(key(s) - target) < tol]
        return max(near, key=lambda s: s[4]) if near else None  # longest

    top_l  = _best_seg(all_h, top_y,  True)
    bot_l  = _best_seg(all_h, bot_y,  True)
    left_l = _best_seg(all_v, left_x, False)
    right_l = _best_seg(all_v, right_x, False)

    # Axis-aligned fallback when representative segments are missing
    if None in (top_l, bot_l, left_l, right_l):
        return np.float32([
            [left_x, top_y], [right_x, top_y],
            [right_x, bot_y], [left_x, bot_y],
        ])

    tl = _line_intersection(_extend_segment(top_l),  _extend_segment(left_l))
    tr = _line_intersection(_extend_segment(top_l),  _extend_segment(right_l))
    br = _line_intersection(_extend_segment(bot_l),  _extend_segment(right_l))
    bl = _line_intersection(_extend_segment(bot_l),  _extend_segment(left_l))

    if None in (tl, tr, br, bl):
        return None

    return np.float32([tl, tr, br, bl])


def _hough_refine(
    warped_gray: np.ndarray,
    corners_orig: np.ndarray,
    M: np.ndarray,
) -> np.ndarray | None:
    """
    Run Hough on the warped image, find the playing grid boundary, and
    map those corners back to original-image space.
    Returns refined corners (shape 4×2, float32) or None if refinement fails.
    """
    all_h, all_v, h_pos, v_pos = _hough_grid_lines(warped_gray)
    ref_warped = _corners_from_hough(all_h, all_v, h_pos, v_pos, float(WARP_SIZE))

    if ref_warped is None:
        return None

    # Sanity: refined grid must cover at least 25 % of the warp in each dimension
    gw = ref_warped[1, 0] - ref_warped[0, 0]
    gh = ref_warped[2, 1] - ref_warped[0, 1]
    if gw < WARP_SIZE * 0.25 or gh < WARP_SIZE * 0.25:
        return None

    M_inv = np.linalg.inv(M)
    refined_orig = cv2.perspectiveTransform(
        ref_warped.reshape(-1, 1, 2), M_inv
    ).reshape(4, 2).astype(np.float32)

    return refined_orig


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 – Direct Hough fallback (no contour quad found)
# ══════════════════════════════════════════════════════════════════════════════

def _direct_hough_corners(gray: np.ndarray) -> np.ndarray | None:
    """Find board corners by running Hough directly on the original image."""
    all_h, all_v, h_pos, v_pos = _hough_grid_lines(gray)
    corners = _corners_from_hough(all_h, all_v, h_pos, v_pos, float(min(gray.shape)))
    return corners


# ══════════════════════════════════════════════════════════════════════════════
# Grid drawing & square data
# ══════════════════════════════════════════════════════════════════════════════

def _grid_intersections() -> np.ndarray:
    """All 9×9 = 81 grid intersection points in warped space, shape (-1,1,2)."""
    sq  = WARP_SIZE / 8.0
    pts = [[j * sq, i * sq] for i in range(9) for j in range(9)]
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def draw_grid_on_original(
    image: np.ndarray,
    M: np.ndarray,
    color: tuple = (0, 230, 0),
    thickness: int = 0,
) -> np.ndarray:
    """
    Draw the 8×8 grid on `image` by inverse-transforming the warped grid
    intersections.  `thickness=0` → auto-scaled to image size.
    """
    h, w = image.shape[:2]
    if thickness == 0:
        thickness = max(2, int(min(h, w) * 0.003))

    M_inv  = np.linalg.inv(M)
    pts    = cv2.perspectiveTransform(_grid_intersections(), M_inv)
    grid   = pts.reshape(9, 9, 2).astype(np.int32)
    out    = image.copy()

    for i in range(9):                  # 9 horizontal lines
        for j in range(8):
            cv2.line(out, tuple(grid[i, j]), tuple(grid[i, j + 1]),
                     color, thickness, cv2.LINE_AA)
    for j in range(9):                  # 9 vertical lines
        for i in range(8):
            cv2.line(out, tuple(grid[i, j]), tuple(grid[i + 1, j]),
                     color, thickness, cv2.LINE_AA)
    return out


def label_squares(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Label each square 0–63 (row-major, top-left = 0)."""
    h, w       = image.shape[:2]
    font_scale = max(0.35, min(h, w) / 2800.0)
    thickness  = max(1, int(font_scale * 2.5))
    M_inv      = np.linalg.inv(M)
    sq         = WARP_SIZE / 8.0
    out        = image.copy()

    for row in range(8):
        for col in range(8):
            pt_w = np.array([[[(col + 0.5) * sq, (row + 0.5) * sq]]],
                            dtype=np.float32)
            pt_o = cv2.perspectiveTransform(pt_w, M_inv)[0, 0].astype(int)
            cv2.putText(out, str(row * 8 + col), tuple(pt_o),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 255), thickness, cv2.LINE_AA)
    return out


def build_squares_list(M: np.ndarray) -> list[dict]:
    """
    Return 64 dicts, each with:
        id      : int  (0 = top-left … 63 = bottom-right, row-major)
        center  : (x, y) in original image coords
        corners : [TL, TR, BR, BL] in original image coords
    """
    M_inv   = np.linalg.inv(M)
    sq      = WARP_SIZE / 8.0
    squares = []

    for row in range(8):
        for col in range(8):
            warped_pts = np.array([
                [col * sq,         row * sq      ],   # TL
                [(col + 1) * sq,   row * sq      ],   # TR
                [(col + 1) * sq,   (row + 1) * sq],   # BR
                [col * sq,         (row + 1) * sq],   # BL
                [(col + 0.5) * sq, (row + 0.5) * sq], # centre
            ], dtype=np.float32).reshape(-1, 1, 2)

            orig_pts = cv2.perspectiveTransform(warped_pts, M_inv).reshape(5, 2)

            squares.append({
                "id":      row * 8 + col,
                "center":  tuple(orig_pts[4].astype(int).tolist()),
                "corners": [tuple(p.astype(int).tolist()) for p in orig_pts[:4]],
            })

    return squares


# ══════════════════════════════════════════════════════════════════════════════
# Debug view
# ══════════════════════════════════════════════════════════════════════════════

def _show_debug(
    rgb: np.ndarray,
    edge_maps: list[np.ndarray],
    corners: np.ndarray,
    warped: np.ndarray,
    method: str,
) -> None:
    labels = ["TL", "TR", "BR", "BL"]
    overlay = rgb.copy()
    pts_int = corners.astype(np.int32)
    cv2.polylines(overlay, [pts_int], True, (255, 50, 50),
                  max(3, min(rgb.shape[:2]) // 150))
    for i, pt in enumerate(pts_int):
        cv2.circle(overlay, tuple(pt), max(10, min(rgb.shape[:2]) // 80),
                   (255, 230, 0), -1)
        cv2.putText(overlay, labels[i], (pt[0] + 12, pt[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, min(rgb.shape[:2]) / 700),
                    (255, 230, 0), 2, cv2.LINE_AA)

    sq = WARP_SIZE // 8
    warp_grid = warped.copy()
    for i in range(9):
        cv2.line(warp_grid, (0, i * sq), (WARP_SIZE, i * sq), (0, 255, 0), 3)
        cv2.line(warp_grid, (i * sq, 0), (i * sq, WARP_SIZE), (0, 255, 0), 3)

    map_titles = ["Edge map A (Otsu+Canny)",
                  "Edge map B (Adaptive+Canny)",
                  "Edge map C (Direct Canny)"]
    n = len(edge_maps)
    fig, axes = plt.subplots(1, n + 2, figsize=(6 * (n + 2), 6))

    axes[0].imshow(overlay)
    axes[0].set_title(f"Detected quad\n({method})", fontsize=10)
    axes[0].axis("off")

    for ax, em, title in zip(axes[1:], edge_maps, map_titles):
        ax.imshow(em, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    axes[-1].imshow(warp_grid)
    axes[-1].set_title("Warped + grid", fontsize=10)
    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Top-level pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_image(
    image_path: str,
    debug: bool = False,
) -> tuple[np.ndarray, list[dict]] | None:
    """
    Full pipeline.
    Returns (annotated_rgb_image, squares_list) or None on failure.
    Pass debug=True to display intermediate edge maps and the detected quad.
    """
    bgr = _load(str(image_path))
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    dst = np.float32([
        [0,         0        ],
        [WARP_SIZE, 0        ],
        [WARP_SIZE, WARP_SIZE],
        [0,         WARP_SIZE],
    ])
    method = "unknown"

    # ── Stage 1: contour quad ─────────────────────────────────────────────────
    best_approx, best_sc = _find_best_contour_quad(gray)

    if best_approx is not None:
        corners = _order_corners(best_approx)
        M       = cv2.getPerspectiveTransform(corners, dst)
        warped  = cv2.warpPerspective(rgb, M, (WARP_SIZE, WARP_SIZE))
        wg      = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

        # ── Stage 2: Hough refinement ─────────────────────────────────────────
        refined = _hough_refine(wg, corners, M)
        if refined is not None:
            corners = refined
            M       = cv2.getPerspectiveTransform(corners, dst)
            warped  = cv2.warpPerspective(rgb, M, (WARP_SIZE, WARP_SIZE))
            method  = "Contour + Hough refinement"
        else:
            method = f"Contour only (score={best_sc:.3f})"

    else:
        # ── Stage 3: direct Hough fallback ────────────────────────────────────
        corners = _direct_hough_corners(gray)
        if corners is None:
            print("[ERROR] Could not detect the chessboard in this image.")
            print("  Tips: ensure the full board is visible, reasonably lit, "
                  "and not cropped at the edges.")
            return None
        corners = _order_corners(corners)
        M       = cv2.getPerspectiveTransform(corners, dst)
        warped  = cv2.warpPerspective(rgb, M, (WARP_SIZE, WARP_SIZE))
        method  = "Direct Hough"

    if debug:
        edge_maps = _build_edge_maps(gray)
        _show_debug(rgb, edge_maps, corners, warped, method)

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = draw_grid_on_original(rgb, M)
    annotated = label_squares(annotated, M)

    dot_r = max(8, int(min(rgb.shape[:2]) * 0.012))
    for pt in corners.astype(np.int32):
        cv2.circle(annotated, tuple(pt), dot_r, (255, 80, 80), -1)

    squares = build_squares_list(M)

    print(f"[OK] {method}")
    return annotated, squares


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    argv  = sys.argv[1:]
    debug = "--debug" in argv
    paths = [a for a in argv if not a.startswith("--")]
    path  = paths[0] if paths else "./test_images/board1.jpg"

    result = process_image(path, debug=debug)
    if result is None:
        sys.exit(1)

    annotated, squares = result

    plt.figure(figsize=(12, 10))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title(f"{Path(path).name}  –  {len(squares)} squares detected")
    plt.tight_layout()
    plt.show()

    print(f"\n{len(squares)} squares detected.")
    for sq in squares[:4]:
        print(f"  id={sq['id']:2d}  center={sq['center']}  corners={sq['corners']}")
    print("  …")


if __name__ == "__main__":
    main()
