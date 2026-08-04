"""Run MediaPipe PoseLandmarker over a video and dump 3D world landmarks.

Usage:
    .venv/bin/python extract_pose.py my_trick.mp4 [-o out/my_trick]

Writes <out>.npz (raw arrays) and <out>.json (for the web viewer).
"""

import argparse
import json
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
# heavy is the most accurate and the default: tricking is fast and self-occluding,
# which is exactly where lite/full start dropping frames.
MODEL_VARIANTS = ("lite", "full", "heavy")
DEFAULT_VARIANT = "heavy"


def model_path(variant=DEFAULT_VARIANT):
    """Path to a PoseLandmarker variant, downloading it on first use."""
    if variant not in MODEL_VARIANTS:
        raise SystemExit(f"unknown model {variant!r}, pick one of {MODEL_VARIANTS}")
    path = os.path.join(MODEL_DIR, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = MODEL_URL.format(v=variant)
        print(f"downloading {variant} model -> {path}")
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
    return path


MODEL_PATH = os.path.join(MODEL_DIR, f"pose_landmarker_{DEFAULT_VARIANT}.task")

# (start, end) index pairs of the 33-point BlazePose skeleton, grouped for coloring.
EDGES = {
    "torso": [(11, 12), (11, 23), (12, 24), (23, 24)],
    "left_arm": [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
    "right_arm": [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
    "left_leg": [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)],
    "right_leg": [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)],
    "face": [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]


def extract(video_path, out_prefix, variant=DEFAULT_VARIANT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_path}: {width}x{height} @ {fps:.2f}fps, {n_frames} frames")
    print(f"model: {variant}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path(variant)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    world = []          # metres, hip-centred -> what we plot in 3D
    screen = []         # normalised image coords, useful for the 2D overlay
    detected = []
    times = []          # true presentation time of each frame, seconds

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        last_ms = -1
        while True:
            # Phone video is often variable frame rate (dropped/duplicated
            # frames), so index/fps is NOT the frame's real time. Read the
            # container's own timestamp before decoding, and key everything off
            # it -- otherwise the viewer's overlay drifts off the athlete.
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = cap.read()
            if not ok:
                break
            if not pos_ms or pos_ms <= last_ms:      # backend gave us nothing usable
                pos_ms = last_ms + 1000.0 / fps
            last_ms = pos_ms
            times.append(pos_ms / 1000.0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(pos_ms))

            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                nl = result.pose_landmarks[0]
                world.append([[p.x, p.y, p.z, p.visibility] for p in wl])
                screen.append([[p.x, p.y, p.z, p.visibility] for p in nl])
                detected.append(True)
            else:
                world.append(np.full((33, 4), np.nan).tolist())
                screen.append(np.full((33, 4), np.nan).tolist())
                detected.append(False)

            idx += 1
            if idx % 30 == 0:
                print(f"  frame {idx}/{n_frames}", end="\r", flush=True)

    cap.release()

    times = np.array(times, dtype=np.float64)
    world = np.array(world, dtype=np.float32)
    screen = np.array(screen, dtype=np.float32)
    detected = np.array(detected)
    print(f"\ndetected pose in {detected.sum()}/{len(detected)} frames")

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.savez_compressed(
        out_prefix + ".npz",
        world=world, screen=screen, detected=detected, times=times,
        fps=fps, width=width, height=height, names=np.array(LANDMARK_NAMES),
    )

    # MediaPipe world coords are y-down, with z increasing away from the camera
    # (origin at the hip midpoint). Flip y only, so the viewer shows the athlete
    # upright while z stays a true camera distance -- negating z as well would
    # invert depth and make the figure spin the wrong way.
    pts = world[:, :, :3].copy()
    pts[:, :, 1] *= -1.0

    # normalised image coords for the 2D overlay drawn on top of the video
    uv = screen[:, :, :2]

    payload = {
        "fps": float(fps),
        "names": LANDMARK_NAMES,
        "edges": EDGES,
        "detected": detected.tolist(),
        # real presentation times; the viewer seeks by these, not index/fps
        "times": np.round(times, 4).tolist(),
        # round to mm; keeps the JSON small enough to inline
        "frames": np.where(np.isnan(pts), None, np.round(pts, 4)).tolist(),
        "screen": np.where(np.isnan(uv), None, np.round(uv, 4)).tolist(),
        "visibility": np.round(np.nan_to_num(world[:, :, 3]), 3).tolist(),
        "source": os.path.basename(video_path),
        "model": variant,
        "video": {"width": width, "height": height, "src": None},
    }
    with open(out_prefix + ".json", "w") as f:
        json.dump(payload, f, separators=(",", ":"), default=lambda o: None)

    print(f"wrote {out_prefix}.npz and {out_prefix}.json")
    return out_prefix + ".json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default=None, help="output prefix, default out/<videoname>")
    ap.add_argument("--embed-video", action="store_true",
                    help="inline the video as a data URI so the .html is fully portable "
                         "(makes the file ~1.4x the video size)")
    ap.add_argument("--no-video", action="store_true", help="skip the 2D overlay panel")
    ap.add_argument("-m", "--model", choices=MODEL_VARIANTS, default=DEFAULT_VARIANT,
                    help="PoseLandmarker variant; heavy (default) is most accurate, "
                         "lite is fastest. Downloaded on first use.")
    args = ap.parse_args()

    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0])
    json_path = extract(args.video, out, variant=args.model)

    from make_viewer import build_viewer
    html = build_viewer(json_path, out + ".html",
                        video=None if args.no_video else args.video,
                        embed=args.embed_video)
    print(f"open {html}")


if __name__ == "__main__":
    main()
