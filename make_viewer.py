"""Bake a pose JSON (+ the source video) into a single self-contained HTML viewer."""

import base64
import json
import mimetypes
import os
import shutil
import subprocess

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "viewer_template.html")

# Codecs Chrome/Safari will play from a plain <video> tag. Phone footage is often
# HEVC or ProRes, which Chrome refuses, so we transcode those to H.264 once.
BROWSER_SAFE = {"h264", "avc1", "vp8", "vp9", "av1"}


def _probe_codec(path):
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip().lower() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _ensure_playable(video, out_dir):
    """Return a path to a video the browser can decode, transcoding if needed."""
    codec = _probe_codec(video)
    if codec is None or codec in BROWSER_SAFE:
        return video
    if not shutil.which("ffmpeg"):
        print(f"warning: video codec is {codec!r}, which browsers may not play, "
              f"and ffmpeg is not installed to convert it")
        return video

    dest = os.path.join(out_dir, os.path.splitext(os.path.basename(video))[0] + "_h264.mp4")
    if os.path.exists(dest) and os.path.getmtime(dest) >= os.path.getmtime(video):
        return dest
    print(f"transcoding {codec} -> h264 for browser playback ...")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", dest],
        check=True,
    )
    return dest


def build_viewer(json_path, out_html, video=None, embed=False):
    with open(json_path) as f:
        data = json.load(f)

    if video and os.path.exists(video):
        out_dir = os.path.dirname(os.path.abspath(out_html)) or "."
        os.makedirs(out_dir, exist_ok=True)
        playable = _ensure_playable(video, out_dir)
        if embed:
            mime = mimetypes.guess_type(playable)[0] or "video/mp4"
            with open(playable, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            data.setdefault("video", {})["src"] = f"data:{mime};base64,{b64}"
        else:
            # relative so the .html keeps working if the folder is moved
            data.setdefault("video", {})["src"] = os.path.relpath(
                os.path.abspath(playable), out_dir)
    elif video:
        print(f"warning: {video} not found, building viewer without the 2D panel")

    with open(TEMPLATE_PATH) as f:
        tpl = f.read()
    title = data.get("source", "pose")
    blob = json.dumps(data, separators=(",", ":"))
    html = tpl.replace("/*__DATA__*/null", blob).replace("__TITLE__", title)
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("-v", "--video", default=None)
    ap.add_argument("--embed-video", action="store_true")
    a = ap.parse_args()
    print(build_viewer(a.json, os.path.splitext(a.json)[0] + ".html",
                       video=a.video, embed=a.embed_video))
