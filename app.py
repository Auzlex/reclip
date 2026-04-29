"""ReClip — lightweight self-hosted video/audio downloader."""

import logging
import os
import re
import time
import uuid
import glob
import json
import subprocess
import threading
from typing import Any
from urllib.parse import urlparse

from flask import Flask, request, jsonify, send_file, render_template, Response

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("reclip")

# ---------------------------------------------------------------------------
app = Flask(__name__)

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

MAX_TITLE_LEN = 80
JOB_TTL_SECONDS = 3600  # auto-purge jobs older than 1 hour
MAX_JOBS = 500  # hard cap to prevent memory exhaustion
DOWNLOAD_TIMEOUT = 1800  # 30 minutes for large files

# Cleanup settings (PR #12) - Configurable via Env Vars
MAX_DOWNLOAD_AGE_HOURS = float(os.environ.get("MAX_DOWNLOAD_AGE_HOURS", 1))
MAX_DOWNLOAD_DIR_SIZE_MB = int(os.environ.get("MAX_DOWNLOAD_DIR_SIZE_MB", 500))

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
_SAFE_FILENAME_RE = re.compile(r'[^\w\s\-\.]', re.UNICODE)
_MULTI_SPACE_RE = re.compile(r'\s+')
_FORMAT_ID_RE = re.compile(r'^[\w\-\+]+$')


def _validate_url(url: str) -> str | None:
    """Return an error message if *url* is not a valid HTTP(S) URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"
    if parsed.scheme not in ("http", "https"):
        return "Only http and https URLs are supported"
    if not parsed.netloc:
        return "Invalid URL"
    return None


def _sanitize_title(title: str) -> str:
    """Produce a filesystem-safe title string."""
    title = title.strip()
    # Remove null bytes and control characters
    title = re.sub(r'[\x00-\x1f\x7f]', '', title)
    title = _SAFE_FILENAME_RE.sub('', title)
    title = _MULTI_SPACE_RE.sub(' ', title).strip()
    # Prevent hidden files
    title = title.lstrip('.')
    return title[:MAX_TITLE_LEN].strip()


def _validate_format_id(format_id: str | None) -> bool:
    """Return True if format_id looks safe (alphanumeric, dashes, plus)."""
    if format_id is None:
        return True
    return bool(_FORMAT_ID_RE.match(format_id))


def _purge_stale_jobs() -> None:
    """Remove completed/errored jobs older than JOB_TTL_SECONDS and their files."""
    now = time.time()
    stale = []
    with jobs_lock:
        for jid, job in jobs.items():
            if job["status"] in ("done", "error") and now - job.get("created", now) > JOB_TTL_SECONDS:
                stale.append(jid)
        for jid in stale:
            _cleanup_job_files(jid)
            del jobs[jid]
    if stale:
        log.info("Purged %d stale job(s)", len(stale))


def _cleanup_job_files(job_id: str) -> None:
    """Delete all files associated with a job."""
    for f in glob.glob(os.path.join(DOWNLOAD_DIR, f"{job_id}.*")):
        try:
            os.remove(f)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Disk cleanup (PR #12)
# ---------------------------------------------------------------------------

def cleanup_old_downloads() -> None:
    """Remove download files older than MAX_DOWNLOAD_AGE_HOURS."""
    now = time.time()
    cutoff = now - (MAX_DOWNLOAD_AGE_HOURS * 3600)
    removed = 0
    for f in glob.glob(os.path.join(DOWNLOAD_DIR, "*")):
        try:
            if os.path.isfile(f) and os.path.getmtime(f) < cutoff:
                os.remove(f)
                removed += 1
        except OSError:
            pass
    if removed:
        log.info("Cleanup: removed %d old download(s)", removed)


def enforce_dir_size_limit() -> None:
    """If downloads dir exceeds MAX_DOWNLOAD_DIR_SIZE_MB, remove oldest files first."""
    files = []
    total = 0
    for f in glob.glob(os.path.join(DOWNLOAD_DIR, "*")):
        if os.path.isfile(f):
            size = os.path.getsize(f)
            total += size
            files.append((os.path.getmtime(f), f, size))

    max_bytes = MAX_DOWNLOAD_DIR_SIZE_MB * 1024 * 1024
    if total <= max_bytes:
        return

    # Sort oldest first, remove until under limit
    files.sort()
    removed = 0
    for _, f, size in files:
        try:
            os.remove(f)
            total -= size
            removed += 1
            if total <= max_bytes:
                break
        except OSError:
            pass
    if removed:
        log.info("Cleanup: removed %d file(s) to enforce size limit", removed)


# ---------------------------------------------------------------------------
# Download worker (PR #13 Popen-based progress tracking + PR #3 security)
# ---------------------------------------------------------------------------

def run_download(job_id: str, url: str, format_choice: str, format_id: str | None) -> None:
    """Execute yt-dlp in a background thread with real-time progress parsing."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return

    out_template = os.path.join(DOWNLOAD_DIR, f"{job_id}.%(ext)s")

    cmd = ["yt-dlp", "--no-playlist", "--no-warnings", "-o", out_template]

    # Enable progress tracking via newline-separated output (PR #13)
    cmd += ["--newline", "--progress"]

    if format_choice == "audio":
        cmd += ["-x", "--audio-format", "mp3"]
    elif format_id:
        cmd += ["-f", f"{format_id}+bestaudio/best", "--merge-output-format", "mp4"]
    else:
        cmd += ["-f", "bestvideo+bestaudio/best", "--merge-output-format", "mp4"]

    cmd.append(url)

    log.info("Job %s: starting download for %s", job_id, url)

    try:
        # Use Popen for real-time progress parsing (PR #13)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        progress_pattern = re.compile(r"\[download\]\s+Destination:|\[download\]\s+(\d+\.?\d*)%")
        last_error_lines: list[str] = []

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            # Track progress percentage
            match = progress_pattern.search(line)
            if match:
                if match.group(1):
                    try:
                        pct = float(match.group(1))
                        with jobs_lock:
                            job["progress"] = min(pct, 100.0)
                    except ValueError:
                        pass

            # Track error/warning lines for better error reporting
            if line.startswith("ERROR:") or line.startswith("WARNING:"):
                last_error_lines.append(line)
                log.warning("Job %s: %s", job_id, line)

        process.wait(timeout=DOWNLOAD_TIMEOUT)

        if process.returncode != 0:
            # Use the most informative error line
            if last_error_lines:
                error_msg = last_error_lines[-1].replace("ERROR: ", "")
            else:
                error_msg = f"yt-dlp exited with code {process.returncode}"
            log.warning("Job %s: yt-dlp failed — %s", job_id, error_msg)
            with jobs_lock:
                job["status"] = "error"
                job["error"] = error_msg
            return

        files = glob.glob(os.path.join(DOWNLOAD_DIR, f"{job_id}.*"))
        if not files:
            with jobs_lock:
                job["status"] = "error"
                job["error"] = "Download completed but no file was found"
            return

        if format_choice == "audio":
            target = [f for f in files if f.endswith(".mp3")]
        else:
            target = [f for f in files if f.endswith(".mp4")]
        chosen = target[0] if target else files[0]

        # Remove intermediate files
        for f in files:
            if f != chosen:
                try:
                    os.remove(f)
                except OSError:
                    pass

        ext = os.path.splitext(chosen)[1]
        title = job.get("title", "")
        safe_title = _sanitize_title(title)
        filename = f"{safe_title}{ext}" if safe_title else os.path.basename(chosen)

        # Report file size (PR #13)
        file_size_mb = None
        try:
            file_size = os.path.getsize(chosen)
            file_size_mb = round(file_size / (1024 * 1024), 2)
        except OSError:
            pass

        with jobs_lock:
            job["status"] = "done"
            job["progress"] = 100.0
            job["file"] = chosen
            job["filename"] = filename
            if file_size_mb is not None:
                job["file_size_mb"] = file_size_mb

        log.info("Job %s: done — %s", job_id, filename)

    except subprocess.TimeoutExpired:
        with jobs_lock:
            job["status"] = "error"
            job["error"] = f"Download timed out ({DOWNLOAD_TIMEOUT // 60} min limit). Try a lower quality format for large files."
        log.warning("Job %s: timed out", job_id)
        # Clean up partial file (PR #13)
        try:
            process.kill()
        except Exception:
            pass
        for f in glob.glob(os.path.join(DOWNLOAD_DIR, f"{job_id}.*")):
            try:
                os.remove(f)
            except OSError:
                pass
    except Exception as e:
        with jobs_lock:
            job["status"] = "error"
            job["error"] = "Internal download error"
        log.exception("Job %s: unexpected error — %s", job_id, e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/cleanup", methods=["POST"])
def cleanup_endpoint() -> Response:
    """Manually trigger cleanup of old downloads (PR #12)."""
    cleanup_old_downloads()
    enforce_dir_size_limit()
    return jsonify({"status": "ok", "message": "Cleanup completed"})


@app.route("/api/downloads/stats", methods=["GET"])
def downloads_stats() -> Response:
    """Return stats about the downloads folder (PR #12)."""
    files = glob.glob(os.path.join(DOWNLOAD_DIR, "*"))
    file_list = []
    total_size = 0
    for f in files:
        if os.path.isfile(f):
            size = os.path.getsize(f)
            total_size += size
            file_list.append({
                "name": os.path.basename(f),
                "size": size,
                "modified": os.path.getmtime(f),
            })
    file_list.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify({
        "count": len(file_list),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files": file_list,
    })


@app.route("/api/info", methods=["POST"])
def get_info() -> tuple[Response, int] | Response:
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    url = (data.get("url") or "").strip()
    url_err = _validate_url(url)
    if url_err:
        return jsonify({"error": url_err}), 400

    cmd = ["yt-dlp", "--no-playlist", "--no-warnings", "-j", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            stderr_last = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            return jsonify({"error": stderr_last}), 400

        info = json.loads(result.stdout)

        # Build quality options — keep best format per resolution
        best_by_height: dict[int, dict] = {}
        for f in info.get("formats", []):
            height = f.get("height")
            if height and f.get("vcodec", "none") != "none":
                tbr = f.get("tbr") or 0
                if height not in best_by_height or tbr > (best_by_height[height].get("tbr") or 0):
                    best_by_height[height] = f

        formats = []
        for height, f in best_by_height.items():
            formats.append({
                "id": f["format_id"],
                "label": f"{height}p",
                "height": height,
            })
        formats.sort(key=lambda x: x["height"], reverse=True)

        return jsonify({
            "title": info.get("title", ""),
            "thumbnail": info.get("thumbnail", ""),
            "duration": info.get("duration"),
            "uploader": info.get("uploader", ""),
            "formats": formats,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timed out fetching video info"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse video info"}), 400
    except Exception:
        log.exception("Error in /api/info")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/download", methods=["POST"])
def start_download() -> tuple[Response, int] | Response:
    _purge_stale_jobs()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    url = (data.get("url") or "").strip()
    url_err = _validate_url(url)
    if url_err:
        return jsonify({"error": url_err}), 400

    format_choice = data.get("format", "video")
    if format_choice not in ("video", "audio"):
        return jsonify({"error": "Invalid format"}), 400

    format_id = data.get("format_id")
    if not _validate_format_id(format_id):
        return jsonify({"error": "Invalid format_id"}), 400

    title = (data.get("title") or "")[:200]  # cap title length from client

    with jobs_lock:
        if len(jobs) >= MAX_JOBS:
            return jsonify({"error": "Server is busy, please try again later"}), 503

    # Run disk cleanup before starting new downloads (PR #12)
    cleanup_old_downloads()
    enforce_dir_size_limit()

    job_id = uuid.uuid4().hex[:10]
    with jobs_lock:
        jobs[job_id] = {
            "status": "downloading",
            "url": url,
            "title": title,
            "created": time.time(),
        }

    thread = threading.Thread(
        target=run_download,
        args=(job_id, url, format_choice, format_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def check_status(job_id: str) -> tuple[Response, int] | Response:
    # Validate job_id format (hex, 10 chars)
    if not re.match(r'^[0-9a-f]{10}$', job_id):
        return jsonify({"error": "Invalid job ID"}), 400

    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "status": job["status"],
        "error": job.get("error"),
        "filename": job.get("filename"),
        "progress": job.get("progress"),
        "file_size_mb": job.get("file_size_mb"),
    })


@app.route("/api/file/<job_id>")
def download_file(job_id: str) -> tuple[Response, int] | Response:
    # Validate job_id format
    if not re.match(r'^[0-9a-f]{10}$', job_id):
        return jsonify({"error": "Invalid job ID"}), 400

    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "File not ready"}), 404

    filepath = job.get("file", "")

    # Path traversal protection: ensure file is inside DOWNLOAD_DIR (PR #3)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(os.path.realpath(DOWNLOAD_DIR)):
        log.warning("Path traversal attempt blocked: %s", filepath)
        return jsonify({"error": "Access denied"}), 403

    if not os.path.isfile(real_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(real_path, as_attachment=True, download_name=job.get("filename", "download"))


@app.route("/api/cleanup/<job_id>", methods=["POST"])
def cleanup_job(job_id: str) -> tuple[Response, int] | Response:
    """Allow clients to signal they've downloaded the file so we can clean up (PR #3)."""
    if not re.match(r'^[0-9a-f]{10}$', job_id):
        return jsonify({"error": "Invalid job ID"}), 400

    with jobs_lock:
        job = jobs.pop(job_id, None)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    _cleanup_job_files(job_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Clean up stale downloads on startup (PR #12)
    cleanup_old_downloads()
    enforce_dir_size_limit()

    port = int(os.environ.get("PORT", 8899))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
