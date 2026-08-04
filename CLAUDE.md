# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A prototype that takes a video of someone tricking, runs MediaPipe PoseLandmarker
over it, and bakes a self-contained HTML viewer: the source video with the
skeleton overlaid on the left, a 3D scene on the right, both driven by one frame
index.

## Commands

There is no test suite, linter, or build step. The venv is committed-adjacent but
gitignored; recreate it with:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mediapipe opencv-python numpy
```

Always invoke `.venv/bin/python` directly — the system Python is 3.9 and
MediaPipe wheels target 3.10+.

```bash
# full pipeline: inference + JSON/npz + viewer
.venv/bin/python extract_pose.py my_trick.mp4

# rebuild only the viewer from an existing JSON (no inference — seconds, not minutes)
.venv/bin/python make_viewer.py out/my_trick.json -v my_trick.mp4

# synthetic fixture: exercises the viewer with no footage and no MediaPipe run
.venv/bin/python examples/make_demo.py
```

`examples/make_demo.py` is the closest thing to a test. Run it after touching
`viewer_template.html`; it builds a spinning stick figure over an ffmpeg test
pattern, so anything that renders wrong is a viewer bug, not a data bug.

Inference on a 6s 60fps clip with the heavy model takes a few minutes. Use
`-m lite` when iterating on extraction logic.

### Viewing the output

The Chrome extension **blocks `file://` URLs**, so screenshotting a viewer
requires a local server:

```bash
nohup .venv/bin/python -m http.server 8730 >/dev/null 2>&1 &
# then open http://localhost:8730/out/my_trick.html
```

When screenshotting a specific frame, drive it from JS and wait on the video's
`seeked` event — a plain `setFrame()` plus a sleep will often capture a stale
video frame and look like a registration bug:

```js
setFrame(174); await new Promise(r => { vid.onseeked = r; setTimeout(r, 2000); });
```

## Architecture

Three files, one data contract.

```
extract_pose.py  →  out/<name>.json  →  make_viewer.py  →  out/<name>.html
                    out/<name>.npz       (+ viewer_template.html)
```

**`extract_pose.py`** — inference plus all the geometry. The pipeline inside
`extract()`, in order:

1. Decode frames, run PoseLandmarker in VIDEO mode. Timestamps come from
   `CAP_PROP_POS_MSEC` read *before* each `read()`, never `index/fps` — phone
   clips are variable frame rate and index-derived time drifts the overlay off
   the athlete.
2. `metric_scale()` — MediaPipe's world landmarks are only loosely metric (~30%
   short on the test clip). Rescales via pose-invariant bone lengths to the
   `--height` the user states.
3. `place_in_scene()` — world landmarks are hip-centred, so the athlete never
   travels. Per-frame `solvePnP` against the observed 2D landmarks recovers the
   transform into the real camera's frame. Focal length comes from the assumed
   `--fov`. Translation is smoothed per-axis with **z three times harder than
   x/y**, because depth is the badly-conditioned axis in a monocular solve.
4. `fit_floor()` — RANSAC over each frame's *lowest foot point* (grounded frames
   land on the plane, airborne ones sit above, feet never go below). Camera
   height is an output of this fit, never an assumption.

**`make_viewer.py`** — inlines the JSON into the template. Also probes the video
codec with `ffprobe` and transcodes non-browser-safe ones (iPhone HEVC, ProRes)
to H.264. Video is referenced by relative path unless `--embed-video`.

**`viewer_template.html`** — the whole viewer, no libraries, no network. Data is
substituted at the `/*__DATA__*/null` marker and `__TITLE__`.

### Coordinate frames (the thing to get right)

Three frames coexist; conflating them is the source of most bugs here.

- **MediaPipe world** — y-down, z increasing away from the camera, hip-centred.
  On export y is flipped and **z is left alone**; negating z inverts depth and
  makes the figure spin the wrong way.
- **`scene` (camera frame)** — metres, camera at the origin looking down +z,
  y-up. This is what the fixed camera projects.
- **World frame (viewer only)** — Y is the fitted ground normal, origin on the
  ground under the athlete. `toWorld()` / `camSpace()` convert. Orbiting happens
  *here*, which is why the ground stays level; rotating in the camera frame makes
  it swing like a ramp.

### Viewer structure

Two camera modes share every draw path via two switches: `PT(i)` picks `scene`
vs hip-centred `frames`, and `project()` dispatches to the fixed rig or the
orbit rig.

- **fixed** (default) — an orbit rig whose *starting* pose is the real camera.
  Constant focal length; scroll dollies along the view ray, shift-drag pans.
  At `rig = RIG_HOME` the transform is an identity, so it reproduces the video's
  framing exactly. **Verify this after touching the camera maths** by
  reprojecting `SCENE` joints against `SCREEN` — median should stay ~10 px.
- **follow** — hip-centred orbit rig; the athlete rotates in place.

Rendering is painter's algorithm — a 2D canvas has no depth buffer, so
primitives are collected with their camera depth and drawn far-to-near. Sizes
must go through `pxPerMetre()`; anything using a hardcoded pixel size stops
behaving like 3D and clumps when you zoom out. Ground grid lines are clipped
against the near plane rather than dropped, otherwise the floor vanishes at
grazing angles.

Canvas colours live in the `PALETTE` object, not CSS — a 2D context can't read
CSS variables. Any new canvas colour needs a light and a dark entry.

## Accuracy caveats to preserve

The metre readouts are estimates and the code should keep saying so (`~` in the
HUD, caveats in the README). Absolute depth wanders ~0.8 m across a clip even
after smoothing, and both `--fov` and `--height` scale distances directly. Pose
*shape* and twist direction are trustworthy; absolute distance is not. Don't
present these numbers as measurements.

## Repo conventions

`models/` (30 MB `.task` files, downloaded on first use), `out/`, and `*.mov`
are gitignored — the repo is public and the footage is personal. Keep it that
way when adding files.
