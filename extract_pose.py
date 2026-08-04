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


def metric_scale(world, athlete_height_m):
    """Scale factor putting the skeleton at a known real height.

    MediaPipe's world landmarks are only approximately metric -- on this footage
    the athlete comes out ~30% short -- so every derived distance inherits that
    error. Summing rigid bone lengths gives a pose-invariant body size we can
    rescale against a stated height. Scaling the object points scales the PnP
    translation with them, so reprojection is unchanged; only the units move.
    """
    chain = [(27, 25), (25, 23), (23, 11), (11, 0)]     # ankle->knee->hip->shoulder->nose
    total = 0.0
    for a, b in chain:
        seg = np.linalg.norm(world[:, a, :3] - world[:, b, :3], axis=1)
        seg = seg[np.isfinite(seg)]
        if not len(seg):
            return 1.0
        total += float(np.median(seg))
    if total <= 1e-6:
        return 1.0
    # that chain spans ankle to nose, about 0.93 of standing height
    return (0.93 * athlete_height_m) / total


def place_in_scene(world, screen, width, height, fov_deg=65.0, smooth=5, scale=1.0):
    """Put the hip-centred world landmarks back where they really were.

    MediaPipe's world landmarks are metric but hip-centred, so on their own the
    athlete rotates in place and never travels. Solving PnP between those 3D
    points and the observed 2D landmarks recovers the rigid transform into the
    real camera's frame, which is what lets us stand a fixed camera exactly
    where the phone was and watch the athlete move through the scene.

    Focal length is unknown for an arbitrary clip, so it comes from an assumed
    horizontal FOV; getting it wrong scales distance from the camera, not the
    pose itself. Returns (scene, intrinsics) with scene in y-up metres.
    """
    T = len(world)
    fx = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx                                   # square pixels
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    rots = [None] * T
    trans = np.full((T, 3), np.nan)

    for i in range(T):
        obj = world[i, :, :3].astype(np.float64) * scale
        vis = world[i, :, 3]
        img = screen[i, :, :2].astype(np.float64) * [width, height]
        ok = np.isfinite(obj).all(1) & np.isfinite(img).all(1) & (vis > 0.5)
        if ok.sum() < 6:
            continue
        try:
            good, rvec, tvec = cv2.solvePnP(
                obj[ok], img[ok], K, None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if not good or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
            continue
        rots[i] = cv2.Rodrigues(rvec)[0]
        trans[i] = tvec.ravel()

    # PnP translation is jittery frame to frame. Depth is by far the worst axis
    # -- a monocular solve barely constrains it -- so smooth z several times
    # harder than x/y, which are pinned down well by the image position.
    if smooth and smooth > 1:
        sm = trans.copy()
        for axis, win_len in ((0, smooth), (1, smooth), (2, smooth * 3)):
            half = win_len // 2
            col = trans[:, axis]
            for i in range(T):
                lo, hi = max(0, i - half), min(T, i + half + 1)
                w = col[lo:hi]
                w = w[np.isfinite(w)]
                if len(w):
                    sm[i, axis] = w.mean()
        trans = sm

    scene = np.full((T, 33, 3), np.nan, dtype=np.float32)
    for i in range(T):
        if rots[i] is None or not np.isfinite(trans[i]).all():
            continue
        pts = (rots[i] @ (world[i, :, :3].astype(np.float64) * scale).T).T + trans[i]
        pts[:, 1] *= -1.0                      # camera y-down -> viewer y-up
        scene[i] = pts

    placed = int(np.isfinite(scene[:, 0, 0]).sum())
    print(f"placed {placed}/{T} frames in the camera's frame "
          f"(assumed {fov_deg:.0f}deg horizontal FOV)")

    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": width, "height": height, "fov": fov_deg}
    return scene, intr


def fit_floor(scene):
    """Find the ground plane under the athlete.

    Fitting a plane to foot positions sounds right but fails: foot *depth* is
    the noisiest quantity we have, so the fit tilts to chase that noise. The
    athlete's own long axis is much better constrained -- it comes from the PnP
    rotation, which the image pins down well -- so take "up" from the torso on
    frames where they are clearly upright, then drop the plane onto the lowest
    foot point.

    Returns (n, d) with |n| = 1, n pointing up, and n.p + d = 0 on the plane.
    """
    ups, spans = [], []
    for f in scene:
        if not np.isfinite(f).all():
            continue
        ankle = (f[27] + f[28]) / 2.0
        shoulder = (f[11] + f[12]) / 2.0
        v = shoulder - ankle
        L = np.linalg.norm(v)
        if L > 1e-6:
            ups.append(v / L)
            spans.append(L)

    if len(ups) < 5:
        return np.array([0.0, 1.0, 0.0]), -float(np.nanmin(scene[:, :, 1]))

    ups, spans = np.asarray(ups), np.asarray(spans)
    # Upright frames are the ones where ankle->shoulder is near its longest;
    # crouched or inverted frames say nothing useful about which way is up.
    tall = spans >= np.percentile(spans, 70)
    n = ups[tall].mean(0)
    n /= np.linalg.norm(n)
    if n[1] < 0:
        n = -n

    # Anchor the plane on the upright frames only. Monocular depth wanders by
    # the better part of a metre across a clip, so a global min would latch onto
    # the single worst-placed frame; standing frames are the trustworthy ones.
    good = scene[np.isfinite(scene).all((1, 2))]
    foot_h = (good @ n)[:, [29, 30, 31, 32]].min(1)
    d = -float(np.median(foot_h[tall])) if tall.sum() else -float(np.median(foot_h))

    tilt = np.rad2deg(np.arccos(np.clip(n[1], -1, 1)))
    print(f"ground plane from {int(tall.sum())} upright frames, "
          f"{tilt:.1f}deg off the image vertical, camera {d:.2f} m above it")
    return n, d


def extract(video_path, out_prefix, variant=DEFAULT_VARIANT, fov=65.0, smooth=5,
            athlete_height=1.75):
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

    # global placement: where the athlete actually was relative to the camera
    scale = metric_scale(world, athlete_height)
    print(f"metric scale x{scale:.2f} (skeleton normalised to a {athlete_height:.2f} m athlete)")
    scene, intr = place_in_scene(world, screen, width, height,
                                 fov_deg=fov, smooth=smooth, scale=scale)
    floor_n, floor_d = fit_floor(scene)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.savez_compressed(
        out_prefix + ".npz",
        world=world, screen=screen, detected=detected, times=times, scene=scene,
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
        "scene": np.where(np.isnan(scene), None, np.round(scene, 4)).tolist(),
        "intrinsics": intr,
        "athlete_height": athlete_height,
        "floor": {"n": [float(v) for v in floor_n], "d": floor_d},
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
    ap.add_argument("--fov", type=float, default=65.0,
                    help="assumed horizontal field of view in degrees, used to place "
                         "the athlete relative to the camera (default 65, typical phone)")
    ap.add_argument("--height", type=float, default=1.75, dest="athlete_height",
                    help="your standing height in metres; MediaPipe's own metric scale "
                         "is unreliable, so distances are normalised to this (default 1.75)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="frames of centred averaging on the recovered global position; "
                         "0 disables (default 5)")
    args = ap.parse_args()

    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0])
    json_path = extract(args.video, out, variant=args.model,
                        fov=args.fov, smooth=args.smooth,
                        athlete_height=args.athlete_height)

    from make_viewer import build_viewer
    html = build_viewer(json_path, out + ".html",
                        video=None if args.no_video else args.video,
                        embed=args.embed_video)
    print(f"open {html}")


if __name__ == "__main__":
    main()
