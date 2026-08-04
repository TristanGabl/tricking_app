"""Build a demo viewer from a synthetic rotating skeleton.

This is the fixture used to check the viewer itself — projection, playback,
video sync, 2D overlay registration — without needing real footage or a
MediaPipe run. The skeleton is a hand-built stick figure spun about the Y axis
with a vertical bob, so both panes should show a figure turning in place.

    .venv/bin/python examples/make_demo.py
    open examples/out/demo.html
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_pose import EDGES, LANDMARK_NAMES  # noqa: E402
from make_viewer import build_viewer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(HERE, "demo_clip.mp4")
OUT_DIR = os.path.join(HERE, "out")
T = 45          # frames, matches demo_clip.mp4
FPS = 30.0

# A crude 33-point figure in the same y-up metre frame extract_pose.py emits.
BASE = np.zeros((33, 3), np.float32)
BASE[0] = [0, .75, .05]                                  # nose
BASE[11], BASE[12] = [-.18, .5, 0], [.18, .5, 0]         # shoulders
BASE[13], BASE[14] = [-.3, .25, 0], [.3, .25, 0]         # elbows
BASE[15], BASE[16] = [-.35, 0, 0], [.35, 0, 0]           # wrists
BASE[23], BASE[24] = [-.12, 0, 0], [.12, 0, 0]           # hips
BASE[25], BASE[26] = [-.13, -.45, 0], [.13, -.45, 0]     # knees
BASE[27], BASE[28] = [-.13, -.9, 0], [.13, -.9, 0]       # ankles
BASE[29], BASE[30] = [-.13, -.95, -.05], [.13, -.95, -.05]   # heels
BASE[31], BASE[32] = [-.13, -.95, .1], [.13, -.95, .1]       # foot index


def ensure_clip():
    """Regenerate demo_clip.mp4 with ffmpeg if it went missing."""
    if os.path.exists(CLIP):
        return
    print("demo_clip.mp4 missing, regenerating with ffmpeg ...")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=640x480:rate={int(FPS)}:duration={T / FPS}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", CLIP],
        check=True,
    )


def main():
    ensure_clip()
    os.makedirs(OUT_DIR, exist_ok=True)

    pts = np.zeros((T, 33, 3), np.float32)
    for t in range(T):
        a = t / T * 2 * np.pi
        rot = np.array([[np.cos(a), 0, np.sin(a)],
                        [0, 1, 0],
                        [-np.sin(a), 0, np.cos(a)]], np.float32)
        pts[t] = BASE @ rot.T
        pts[t][:, 1] += .3 * np.sin(a)

    # fake normalised image coords: orthographic drop of the 3D points
    uv = np.stack([pts[:, :, 0] * 0.5 + 0.5, 0.5 - pts[:, :, 1] * 0.4], axis=-1)

    payload = {
        "fps": FPS,
        "names": LANDMARK_NAMES,
        "edges": EDGES,
        "detected": [True] * T,
        "frames": np.round(pts, 4).tolist(),
        "screen": np.round(uv, 4).tolist(),
        "visibility": np.ones((T, 33)).tolist(),
        "source": "demo_clip.mp4",
        "video": {"width": 640, "height": 480, "src": None},
    }
    json_path = os.path.join(OUT_DIR, "demo.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    html = build_viewer(json_path, os.path.join(OUT_DIR, "demo.html"), video=CLIP)
    print(f"open {html}")


if __name__ == "__main__":
    main()
