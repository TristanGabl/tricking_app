# Tricking 3D pose prototype

MediaPipe PoseLandmarker (heavy) → 3D world landmarks → self-contained interactive
web viewer.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mediapipe opencv-python numpy
```

The `.task` model files are not in the repo (30MB each) — they download
automatically into `models/` the first time you use a variant.

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

Pick the model with `-m/--model`:

```bash
.venv/bin/python extract_pose.py my_trick.mp4 -m heavy   # default, most accurate
.venv/bin/python extract_pose.py my_trick.mp4 -m full
.venv/bin/python extract_pose.py my_trick.mp4 -m lite    # fastest, drops more frames
```

Heavy is the default on purpose: tricking is fast and self-occluding, exactly
where lite and full start losing the pose. The variant used is recorded in the
JSON and shown in the viewer's footer.

Re-render the viewer without re-running inference:

```bash
.venv/bin/python make_viewer.py out/my_trick.json -v my_trick.mp4
```

## Viewer controls

The 3D pane **opens at the source camera's own viewpoint**, so it starts as a 3D
rebuild of exactly what the video shows — orbit away to inspect depth, and
"reset view" returns there. The skeleton is drawn with volume (capsule limbs,
sphere joints, a filled torso) under a single key light plus fill, painted
far-to-near, with a contact shadow on the floor so you can read height off the
ground. "solid" toggles back to flat wireframe.

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

- **Phone clips are often variable frame rate.** The extractor records each
  frame's true presentation timestamp and the viewer seeks by those, not by
  `index / fps` — without that the overlay slowly slides off the athlete.
- **World landmarks are hip-centred**, so global travel across the mat is *not*
  in the 3D plot — the athlete rotates in place. Absolute displacement would
  need the 2D `screen` coords plus a camera model.
- Depth (z) is the weakest axis in a single-camera model. Twist direction reads
  fine; exact limb depth during fast rotation does not.
- Fast spins blur frames and drop detections. Shoot 60fps+ if you can; check the
  "detected pose in N/M frames" line and the red no-detection banner in the viewer.
- Try `-m lite` for a fast first pass while framing a clip, then re-run with
  the default heavy for the version you actually study.
