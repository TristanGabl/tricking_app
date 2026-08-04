# Tricking 3D pose prototype

MediaPipe PoseLandmarker (heavy) → 3D world landmarks → self-contained interactive
web viewer.

## Setup (already done)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mediapipe opencv-python numpy
# model already at models/pose_landmarker_heavy.task
```

## Run

```bash
.venv/bin/python extract_pose.py my_trick.mp4
open out/my_trick.html
```

Produces in `out/`:

- `my_trick.npz` — `world` (T,33,4: x,y,z,visibility in metres, hip-centred),
  `screen` (normalised image coords), `detected`, `fps`, `width`, `height`, `names`
- `my_trick.json` — y-up 3D points + normalised 2D points for the viewer
- `my_trick.html` — standalone viewer, pose data inlined

The viewer is **two panes**: your video on the left with the skeleton drawn over
it, the 3D world landmarks on the right. Both are driven by one frame index, so
scrubbing or playing moves them together — the fastest way to spot where
tracking lost you.

By default the HTML points at the video by relative path, so keep the `.html`
and the video in the same relative spot. For one file you can send anywhere:

```bash
.venv/bin/python extract_pose.py my_trick.mp4 --embed-video   # inlines as data URI
.venv/bin/python extract_pose.py my_trick.mp4 --no-video      # 3D pane only
```

Non-browser codecs (iPhone HEVC, ProRes) are detected via `ffprobe` and
transcoded to H.264 into `out/` automatically.

Re-render the viewer without re-running inference:

```bash
.venv/bin/python make_viewer.py out/my_trick.json -v my_trick.mp4
```

## Viewer controls

In the 3D pane: drag = orbit · shift-drag = pan · scroll = zoom. Globally:
space = play/pause · ←/→ = step one frame. Trail dropdown draws the flight path
of a joint (default hip centre — good for reading the arc of a kick or twist).
"ghosts" overlays ±4/±8 frames so you can see rotation direction; "2D overlay"
toggles the skeleton off the video so you can watch the raw footage.

Playback uses the `<video>` element as the master clock when a video is loaded,
so the overlay can never drift out of sync with the frame it was computed from.

## Demo fixture

`examples/` holds a synthetic case for checking the viewer itself — projection,
playback, video sync, overlay registration — with no footage or MediaPipe run
needed. A hand-built stick figure spins about the Y axis over `demo_clip.mp4`,
so both panes should show a figure turning in place.

```bash
.venv/bin/python examples/make_demo.py
open examples/out/demo.html
```

`demo_clip.mp4` is an ffmpeg test pattern; `make_demo.py` regenerates it if it
goes missing. Useful as a before/after check whenever you change
`viewer_template.html`.

## Things to know for tricking footage

- **World landmarks are hip-centred**, so global travel across the mat is *not*
  in the 3D plot — the athlete rotates in place. Absolute displacement would
  need the 2D `screen` coords plus a camera model.
- Depth (z) is the weakest axis in a single-camera model. Twist direction reads
  fine; exact limb depth during fast rotation does not.
- Fast spins blur frames and drop detections. Shoot 60fps+ if you can; check the
  "detected pose in N/M frames" line and the red no-detection banner in the viewer.
- `pose_landmarker_heavy.task` is the most accurate of the three; swap the
  filename in `MODEL_PATH` for `_full` or `_lite` if you want speed.
