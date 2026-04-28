"""
chess_video.py  –  real-time chessboard detection on webcam or video file

Displays three windows side-by-side:
  • Original  – raw feed with green grid overlay + red corner dots
  • Warped    – bird's-eye perspective-corrected board
  • Bird's-eye grid – warped board with 8×8 grid lines drawn

Detection strategy
──────────────────
Phase 1 – Searching (0 → LOCK_IN_SECONDS):
    Detect corners every DETECT_INTERVAL frames, smooth with an EMA to damp
    hand-movement jitter.  A countdown bar on-screen shows time remaining.

Phase 2 – Locked (after LOCK_IN_SECONDS of successful detections):
    Corners are frozen.  Detection stops entirely — only warpPerspective runs
    each frame, so CPU usage drops dramatically.
    Press 'l' to unlock and restart the search phase at any time.

Usage
─────
    # Webcam (default device 0)
    python chess_video.py

    # Specific webcam index
    python chess_video.py --cam 1

    # Video file
    python chess_video.py --video path/to/video.mp4

    # Change detection interval (frames between full re-detections)
    python chess_video.py --interval 20

    # Downscale factor for detection (speeds up processing, 0.5 = half res)
    python chess_video.py --scale 0.5

Keyboard controls
─────────────────
    q / ESC   – quit
    space     – force immediate re-detection on next frame
    l         – unlock (restart search phase, discard locked corners)
    s         – save current frame triple to ./captures/
    p         – pause / unpause
"""

from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# ── Import the shared detection logic ────────────────────────────────────────
# perspective_correction.py must be in the same directory (or on sys.path).
try:
    import perspective_correction as pc
except ImportError:
    sys.exit(
        "[ERROR] perspective_correction.py not found.\n"
        "  Place chess_video.py in the same directory as perspective_correction.py."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DETECT_INTERVAL  = 15      # run full detection every N frames
LOCK_IN_SECONDS  = 5.0     # freeze corners after this many seconds of success
WARP_DISPLAY_SZ  = 480     # warped window display size (square, px)
EMA_ALPHA_FAST   = 0.60    # blend weight after a fresh detection  (higher = snappier)
EMA_ALPHA_SLOW   = 0.15    # blend weight between detections       (lower = smoother)
DETECTION_SCALE  = 0.5     # downscale factor for detection only (1.0 = full res)
OVERLAY_COLOR        = (0, 230, 0)
CORNER_COLOR         = (50, 100, 255)
CORNER_COLOR_LOCKED  = (0, 215, 255)   # gold — indicates locked state
STATUS_COLOR_OK      = (0, 200, 0)
STATUS_COLOR_LOCKED  = (0, 215, 255)   # gold
STATUS_COLOR_BAD     = (0, 60, 220)

DST_PTS = np.float32([
    [0,               0              ],
    [pc.WARP_SIZE,    0              ],
    [pc.WARP_SIZE,    pc.WARP_SIZE   ],
    [0,               pc.WARP_SIZE   ],
])


# ══════════════════════════════════════════════════════════════════════════════
# Corner smoother
# ══════════════════════════════════════════════════════════════════════════════

class CornerSmoother:
    """
    Exponential moving average over the 4 board corners.
    Applies a fast blend immediately after a detection, then a slow blend
    on subsequent frames to damp hand-movement jitter.
    """

    def __init__(self, alpha_fast: float = EMA_ALPHA_FAST,
                 alpha_slow: float = EMA_ALPHA_SLOW) -> None:
        self.alpha_fast  = alpha_fast
        self.alpha_slow  = alpha_slow
        self._smoothed:  np.ndarray | None = None
        self._just_detected = False

    def update_detection(self, corners: np.ndarray) -> None:
        """Call when a new detection is available (shape 4×2)."""
        corners = corners.reshape(4, 2).astype(np.float32)
        if self._smoothed is None:
            self._smoothed = corners.copy()
        else:
            # Match new corners to existing ones to avoid flip artefacts
            corners = _match_corner_order(corners, self._smoothed)
            self._smoothed = (self.alpha_fast * corners
                              + (1.0 - self.alpha_fast) * self._smoothed)
        self._just_detected = True

    def step(self) -> np.ndarray | None:
        """
        Call every frame.  Applies slow EMA between detections.
        Returns current smoothed corners (4×2 float32) or None if no detection yet.
        """
        if self._smoothed is None:
            return None
        if self._just_detected:
            self._just_detected = False          # fast blend already applied
        else:
            # Between detections: hold corners with a very slow drift — effectively
            # a no-op for a stable camera but smooths minor residual jitter.
            pass                                 # nothing: just hold last value
        return self._smoothed.copy()


def _match_corner_order(new: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Reorder `new` corners (4×2) to minimise total distance to `ref` corners.
    Prevents sudden diagonal swaps when a corner briefly disappears.
    Tries all 4 cyclic rotations (the shape is always a convex quad so
    reflections are not considered).
    """
    best_order = new
    best_dist  = np.sum((new - ref) ** 2)
    for k in range(1, 4):
        rotated = np.roll(new, k, axis=0)
        dist    = np.sum((rotated - ref) ** 2)
        if dist < best_dist:
            best_dist  = dist
            best_order = rotated
    return best_order


# ══════════════════════════════════════════════════════════════════════════════
# Detection (runs on downscaled frame)
# ══════════════════════════════════════════════════════════════════════════════

def detect_corners(bgr: np.ndarray, scale: float) -> np.ndarray | None:
    """
    Run the full Stage-1 + Stage-2 + Stage-3 pipeline on a (possibly
    downscaled) BGR frame.  Returns corners in *original* frame coordinates,
    or None if detection failed.
    """
    h, w = bgr.shape[:2]

    if scale < 1.0:
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    else:
        small = bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape

    # Stage 1 – contour quad
    best_approx, best_sc = pc._find_best_contour_quad(gray)

    corners_small: np.ndarray | None = None

    if best_approx is not None:
        corners_s = pc._order_corners(best_approx)
        M_s       = cv2.getPerspectiveTransform(corners_s, DST_PTS)
        warped_s  = cv2.warpPerspective(gray, M_s, (pc.WARP_SIZE, pc.WARP_SIZE))

        # Stage 2 – Hough refinement inside warped
        refined = pc._hough_refine(warped_s, corners_s, M_s)
        corners_small = refined if refined is not None else corners_s

    else:
        # Stage 3 – direct Hough on original
        corners_small = pc._direct_hough_corners(gray)
        if corners_small is not None:
            corners_small = pc._order_corners(corners_small)

    if corners_small is None:
        return None

    # Scale corners back to full-frame coordinates
    if scale < 1.0:
        corners_small = corners_small * np.float32([w / sw, h / sh])

    return corners_small.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def _grid_intersections_cached() -> np.ndarray:
    """Return 81 grid-intersection points in warped space (shape -1,1,2)."""
    sq  = pc.WARP_SIZE / 8.0
    pts = [[j * sq, i * sq] for i in range(9) for j in range(9)]
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

_GRID_PTS = _grid_intersections_cached()   # computed once


def render_overlay(bgr: np.ndarray, corners: np.ndarray, M: np.ndarray,
                   status: str, fps: float) -> np.ndarray:
    """Draw grid + corner dots + HUD text onto a copy of `bgr`."""
    out   = bgr.copy()
    h, w  = out.shape[:2]
    thick = max(1, int(min(h, w) * 0.003))
    dot_r = max(5, int(min(h, w) * 0.010))

    # Grid lines (inverse-transform from warped space)
    M_inv = np.linalg.inv(M)
    pts   = cv2.perspectiveTransform(_GRID_PTS, M_inv)
    grid  = pts.reshape(9, 9, 2).astype(np.int32)

    for i in range(9):
        for j in range(8):
            cv2.line(out, tuple(grid[i, j]), tuple(grid[i, j + 1]),
                     OVERLAY_COLOR, thick, cv2.LINE_AA)
    for j in range(9):
        for i in range(8):
            cv2.line(out, tuple(grid[i, j]), tuple(grid[i + 1, j]),
                     OVERLAY_COLOR, thick, cv2.LINE_AA)

    # Corner dots
    for pt in corners.astype(np.int32):
        cv2.circle(out, tuple(pt), dot_r, CORNER_COLOR, -1, cv2.LINE_AA)

    # HUD
    col  = STATUS_COLOR_OK if status.startswith("OK") else STATUS_COLOR_BAD
    cv2.putText(out, status,   (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    cv2.putText(out, f"{fps:.1f} fps", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_warped(bgr: np.ndarray, corners: np.ndarray,
                  display_size: int = WARP_DISPLAY_SZ) -> np.ndarray:
    """Return the perspective-corrected board as a square BGR image."""
    M      = cv2.getPerspectiveTransform(corners, DST_PTS)
    warped = cv2.warpPerspective(bgr, M, (pc.WARP_SIZE, pc.WARP_SIZE))
    return cv2.resize(warped, (display_size, display_size))


def render_warped_grid(warped_small: np.ndarray,
                       display_size: int = WARP_DISPLAY_SZ) -> np.ndarray:
    """Draw the 8×8 grid on an already-warped (square) image."""
    out = warped_small.copy()
    sq  = display_size // 8
    for i in range(9):
        cv2.line(out, (0, i * sq), (display_size, i * sq), OVERLAY_COLOR, 1, cv2.LINE_AA)
        cv2.line(out, (i * sq, 0), (i * sq, display_size), OVERLAY_COLOR, 1, cv2.LINE_AA)
    return out


def build_display(original: np.ndarray, warped: np.ndarray,
                  warped_grid: np.ndarray, target_h: int = 480) -> np.ndarray:
    """
    Resize all three panels to the same height and concatenate side-by-side.
    """
    def fit_height(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    orig_r  = fit_height(original,    target_h)
    warp_r  = fit_height(warped,      target_h)
    wgrid_r = fit_height(warped_grid, target_h)

    # Separator bars
    sep = np.zeros((target_h, 4, 3), dtype=np.uint8)
    return np.hstack([orig_r, sep, warp_r, sep, wgrid_r])


# ══════════════════════════════════════════════════════════════════════════════
# Save helper
# ══════════════════════════════════════════════════════════════════════════════

def save_capture(display: np.ndarray) -> None:
    out_dir = Path("captures")
    out_dir.mkdir(exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"chess_{ts}.jpg"
    cv2.imwrite(str(path), display)
    print(f"[saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run(source: int | str,
        detect_interval: int = DETECT_INTERVAL,
        scale: float = DETECTION_SCALE,
        lock_in_seconds: float = LOCK_IN_SECONDS) -> None:

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video source: {source!r}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Source: {source}  native FPS: {native_fps:.1f}")
    print(f"[INFO] Will lock corners after {lock_in_seconds:.0f}s of successful detections.")
    print("[INFO] Controls: q/ESC=quit  space=force detect  l=unlock  s=save  p=pause")

    smoother      = CornerSmoother()
    frame_idx     = 0
    force_detect  = True
    paused        = False
    status        = "Searching…"
    last_M: np.ndarray | None = None

    # ── Lock-in state ─────────────────────────────────────────────────────────
    locked          = False          # True once corners are frozen
    locked_corners: np.ndarray | None = None
    locked_M:       np.ndarray | None = None
    search_start    = time.perf_counter()   # when current search phase began
    first_success_t: float | None = None   # time of first successful detection this phase

    # FPS measurement
    fps_times: deque[float] = deque(maxlen=30)
    fps_display = 0.0

    placeholder_warp = np.zeros((WARP_DISPLAY_SZ, WARP_DISPLAY_SZ, 3), dtype=np.uint8)
    cv2.putText(placeholder_warp, "Detecting…", (60, WARP_DISPLAY_SZ // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    while True:
        # ── Keyboard ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            force_detect = True
        if key == ord('l'):
            # Unlock: discard frozen corners, restart search phase
            locked          = False
            locked_corners  = None
            locked_M        = None
            first_success_t = None
            search_start    = time.perf_counter()
            smoother        = CornerSmoother()
            force_detect    = True
            status          = "Searching…"
            print("[INFO] Unlocked — restarting search.")
        if key == ord('p'):
            paused = not paused
            print("[INFO]", "Paused" if paused else "Resumed")

        if paused:
            cv2.waitKey(50)
            continue

        # ── Grab frame ───────────────────────────────────────────────────────
        ret, bgr = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        t0  = time.perf_counter()
        now = t0

        # ── Detection (skipped entirely once locked) ──────────────────────────
        if not locked:
            run_detect = force_detect or (frame_idx % detect_interval == 0)
            force_detect = False

            if run_detect:
                new_corners = detect_corners(bgr, scale)
                if new_corners is not None:
                    smoother.update_detection(new_corners)
                    if first_success_t is None:
                        first_success_t = now
                    elapsed_good = now - first_success_t
                    remaining    = lock_in_seconds - elapsed_good

                    if remaining <= 0:
                        # ── LOCK IN ───────────────────────────────────────────
                        locked_corners = smoother.step()
                        locked_M       = cv2.getPerspectiveTransform(
                                             locked_corners, DST_PTS)
                        locked         = True
                        status         = "LOCKED"
                        print(f"[INFO] Corners locked after "
                              f"{elapsed_good:.1f}s of successful detections.")
                    else:
                        status = f"Searching… lock in {remaining:.1f}s"
                else:
                    # Detection failed — reset the success timer
                    first_success_t = None
                    status          = "No board detected"

            corners = smoother.step()
        else:
            corners = locked_corners

        # ── Render ───────────────────────────────────────────────────────────
        if corners is not None:
            try:
                M      = locked_M if locked else cv2.getPerspectiveTransform(corners, DST_PTS)
                last_M = M

                # Choose corner dot colour: gold when locked, blue while searching
                dot_col = CORNER_COLOR_LOCKED if locked else CORNER_COLOR

                # Draw countdown bar during search phase
                h_frame, w_frame = bgr.shape[:2]
                overlay = bgr.copy()

                if not locked and first_success_t is not None:
                    elapsed_good = now - first_success_t
                    frac         = min(elapsed_good / lock_in_seconds, 1.0)
                    bar_w        = int(w_frame * frac)
                    bar_h        = 8
                    cv2.rectangle(overlay, (0, h_frame - bar_h),
                                  (bar_w, h_frame), (0, 215, 255), -1)

                # Grid lines
                M_inv = np.linalg.inv(M)
                pts   = cv2.perspectiveTransform(_GRID_PTS, M_inv)
                grid  = pts.reshape(9, 9, 2).astype(np.int32)
                thick = max(1, int(min(h_frame, w_frame) * 0.003))
                for i in range(9):
                    for j in range(8):
                        cv2.line(overlay, tuple(grid[i, j]), tuple(grid[i, j+1]),
                                 OVERLAY_COLOR, thick, cv2.LINE_AA)
                for j in range(9):
                    for i in range(8):
                        cv2.line(overlay, tuple(grid[i, j]), tuple(grid[i+1, j]),
                                 OVERLAY_COLOR, thick, cv2.LINE_AA)

                # Corner dots
                dot_r = max(5, int(min(h_frame, w_frame) * 0.010))
                for pt in corners.astype(np.int32):
                    cv2.circle(overlay, tuple(pt), dot_r, dot_col, -1, cv2.LINE_AA)

                # Status text
                s_col = STATUS_COLOR_LOCKED if locked else STATUS_COLOR_OK
                cv2.putText(overlay, status, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_col, 2, cv2.LINE_AA)
                if locked:
                    cv2.putText(overlay, "press L to unlock", (10, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                STATUS_COLOR_LOCKED, 1, cv2.LINE_AA)
                else:
                    cv2.putText(overlay, f"{fps_display:.1f} fps", (10, 54),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 200), 1, cv2.LINE_AA)

                warped = render_warped(bgr, corners)
                wgrid  = render_warped_grid(warped)

            except cv2.error:
                overlay = bgr.copy()
                warped  = placeholder_warp.copy()
                wgrid   = placeholder_warp.copy()
                cv2.putText(overlay, "Bad corners – redetecting", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATUS_COLOR_BAD, 2)
                if locked:
                    locked = False; locked_corners = None; locked_M = None
                    first_success_t = None; search_start = time.perf_counter()
                force_detect = True
        else:
            overlay = bgr.copy()
            cv2.putText(overlay, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATUS_COLOR_BAD, 2, cv2.LINE_AA)
            warped = placeholder_warp.copy()
            wgrid  = placeholder_warp.copy()

        # ── Save ─────────────────────────────────────────────────────────────
        if key == ord('s'):
            display = build_display(overlay, warped, wgrid)
            save_capture(display)

        # ── Show ─────────────────────────────────────────────────────────────
        display = build_display(overlay, warped, wgrid)
        cv2.imshow(
            "Chess Board Detection  [q=quit  space=detect  l=unlock  s=save  p=pause]",
            display,
        )

        # ── FPS ──────────────────────────────────────────────────────────────
        fps_times.append(time.perf_counter() - t0)
        if len(fps_times) == fps_times.maxlen:
            fps_display = 1.0 / (sum(fps_times) / len(fps_times))

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time chessboard detection (webcam or video file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--cam",   type=int,   default=None,
                     metavar="INDEX", help="Webcam device index (default 0)")
    src.add_argument("--video", type=str,   default=None,
                     metavar="PATH",  help="Path to a video file")
    p.add_argument("--interval", type=int,   default=DETECT_INTERVAL,
                   metavar="N", help=f"Frames between detections (default {DETECT_INTERVAL})")
    p.add_argument("--scale",    type=float, default=DETECTION_SCALE,
                   metavar="F", help=f"Downscale factor for detection (default {DETECTION_SCALE})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.video is not None:
        if not Path(args.video).exists():
            sys.exit(f"[ERROR] Video file not found: {args.video}")
        source: int | str = args.video
    else:
        source = args.cam if args.cam is not None else 0

    run(source, detect_interval=args.interval, scale=args.scale)


if __name__ == "__main__":
    main()
