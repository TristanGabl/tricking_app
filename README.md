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
it, a 3D scene on the right. Both are driven by one frame index, so scrubbing or
playing moves them together — the fastest way to spot where tracking lost you.

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

## Camera modes

**fixed — real camera** (default). The camera does not move: it stands exactly
where your phone stood, using the recovered pose and the assumed FOV, and the
athlete travels through the scene in front of it. There is a ground plane, its
horizon, a sky, a key light with a visible sun, and a shadow cast onto the
ground along the light direction. Orbit and zoom are disabled here by design.

Nothing assumes where the camera was — hip height, head height, propped on a
bag or sitting in the grass all work. The ground plane is fitted only to the
lowest foot point of each frame (grounded frames land on it, airborne frames
sit above, feet never go below), and the camera's height above the ground falls
out of that fit. On the test clip it came out 0.13 m up and tilted 17°, i.e. a
phone lying near the grass, which is what that footage is.

**follow — orbit.** The old rig: hip-centred, so the athlete rotates in place,
and you can drag to orbit, shift-drag to pan, scroll to zoom. Use it to inspect
a pose from angles the real camera never had. "reset view" re-centres it.

The skeleton is drawn with volume (capsule limbs, sphere joints, a filled torso)
under a key light plus fill, painted far-to-near. "solid" toggles back to flat
wireframe; "scene" toggles the floor and sky.

**Light and dark** are both supported and the default follows your OS setting.
The theme button overrides it for that session; if you never touch it, the page
keeps following the system live.

## Viewer controls

Globally:
space = play/pause · ←/→ = step one frame. Trail dropdown draws the flight path
of a joint (off by default; hip centre is good for reading the arc of a kick or
twist, but only the last ~45 frames are drawn — see the depth caveat below).
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
- **The metre readouts are estimates, not measurements.** Placing the athlete in
  the camera's frame needs a focal length we don't have, so it comes from an
  assumed `--fov` (65deg). MediaPipe's own metric scale is also unreliable — on
  the test clip the skeleton came out ~30% short — so everything is renormalised
  to `--height` (default 1.75 m). Get either wrong and distances scale with it.
- **Depth is the weak axis.** On the test clip the recovered distance from the
  camera wanders by ~0.8 m across the clip even after smoothing, which is why
  the trail only shows the recent past and why heights are shown with a `~`.
  Pose *shape* and twist direction are solid; absolute depth is not.
- In follow mode the landmarks are hip-centred, so the athlete rotates in place
  and does not travel — that view is about the pose, not the path.
- Fast spins blur frames and drop detections. Shoot 60fps+ if you can; check the
  "detected pose in N/M frames" line and the red no-detection banner in the viewer.
- Try `-m lite` for a fast first pass while framing a clip, then re-run with
  the default heavy for the version you actually study.
