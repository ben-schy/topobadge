"""Local web UI: upload a GPX file, fetch data for the area once, then
interactively adjust colors/widths/water-source choices against an instant
local preview, and only pay the network-fetch cost again if you change the
working area itself. Runs entirely on 127.0.0.1 - single local user, no auth
needed, and every job's files live in their own directory under the system
temp dir so nothing pollutes the user's project.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .gpx import load_point_area, load_track
from .pipeline import (
    LAYER_LABELS,
    SOURCE_CLASS_LABELS,
    AdjustOptions,
    FetchedData,
    FetchOptions,
    LayerSpec,
    build_plaque_mask,
    compose_masks,
    default_layer_assignment,
    default_layers,
    fetch_stage,
    mask_area_percentages,
    mesh_stage,
    render_preview_for,
)

_JOBS_ROOT = os.path.join(tempfile.gettempdir(), "topobadge_jobs")


class Job:
    def __init__(self, job_id: str, job_dir: str) -> None:
        self.id = job_id
        self.dir = job_dir
        self.state = "fetching"  # "fetching" | "ready" | "meshing" | "done" | "error"
        self.log: list[str] = []
        self.error: str | None = None
        self.fetched: FetchedData | None = None
        self.landcover_warning: str | None = None
        self.files: list[str] = []
        self.preview_file: str | None = None
        self.parts: list[dict] = []
        self.started_at = time.time()
        self._lock = threading.Lock()

    def append_log(self, line: str) -> None:
        with self._lock:
            self.log.append(line)

    def mark_file_available(self, filename: str) -> None:
        with self._lock:
            if filename not in self.files:
                self.files.append(filename)

    def snapshot(self, since: int) -> dict:
        with self._lock:
            log_slice = self.log[since:]
            next_cursor = len(self.log)

        fetch_info = None
        if self.fetched is not None:
            dem = self.fetched.dem
            fetch_info = {
                "elevation_min": round(float(dem.min())),
                "elevation_max": round(float(dem.max())),
                "grid_cols": self.fetched.grid.ncols,
                "grid_rows": self.fetched.grid.nrows,
                "warnings": self.fetched.warnings,
            }

        return {
            "state": self.state,
            "log": log_slice,
            "next_cursor": next_cursor,
            "error": self.error,
            "landcover_warning": self.landcover_warning,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "fetch_info": fetch_info,
            "result": None
            if self.state != "done"
            else {"files": self.files, "preview_file": self.preview_file, "parts": self.parts},
        }


_JOBS: dict[str, Job] = {}


def _bool_field(form, key: str, default: bool) -> bool:
    val = form.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "on", "yes")


def _fetch_options_from_form(form) -> FetchOptions:
    return FetchOptions(
        buffer_km=float(form.get("buffer_km", 1.2)),
        size_mm=float(form.get("size_mm", 80.0)),
        mm_per_cell=float(form.get("mm_per_cell", 0.8)),
        base_shape=form.get("base_shape", "hexagon"),
    )


def _adjust_options_from_form(form) -> AdjustOptions:
    layers = default_layers()
    layer_assignment = default_layer_assignment()
    layers_json = form.get("layers_json")
    if layers_json:
        overrides = json.loads(layers_json)
        for name, spec in overrides.get("layers", {}).items():
            layers[name] = LayerSpec(color_hex=spec["color_hex"], rise_mm=float(spec.get("rise_mm", 0.0)))
        layer_assignment.update(overrides.get("layer_assignment", {}))

    return AdjustOptions(
        vertical_exaggeration=float(form.get("vertical_exaggeration", 2.0)),
        base_thickness_mm=float(form.get("base_thickness_mm", 3.0)),
        inlay_depth_mm=float(form.get("inlay_depth_mm", 0.8)),
        trail_width_mm=float(form.get("trail_width_mm", 3.0)),
        water_width_mm=float(form.get("water_width_mm", 2.0)),
        river_width_mm=float(form.get("river_width_mm", 4.0)),
        road_width_mm=float(form.get("road_width_mm", 2.0)),
        other_trail_width_mm=float(form.get("other_trail_width_mm", 2.0)),
        landmark_radius_mm=float(form.get("landmark_radius_mm", 2.5)),
        smoothing=float(form.get("smoothing", 1.0)),
        use_vector_water=_bool_field(form, "use_vector_water", True),
        use_nlcd_water=_bool_field(form, "use_nlcd_water", True),
        layers=layers,
        layer_assignment=layer_assignment,
        plaque_text=(form.get("plaque_text", "") or "").replace("\\n", "\n"),
        plaque_width_mm=float(form.get("plaque_width_mm", 50.0)),
        plaque_depth_mm=float(form.get("plaque_depth_mm", 0.8)),
        plaque_color_hex=form.get("plaque_color_hex", "8B5A2B"),
    )


def _run_fetch(job: Job, gpx_path: str, options: FetchOptions) -> None:
    try:
        job.append_log(f"Loading GPX track: {gpx_path}")
        track = load_track(gpx_path, buffer_km=options.buffer_km)
        job.fetched = fetch_stage(track, options, on_log=job.append_log)
        job.state = "ready"
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI instead of crashing the thread silently
        job.error = str(e)
        job.state = "error"


def _run_fetch_area(job: Job, lat: float, lon: float, options: FetchOptions) -> None:
    """Map-picker mode: the working area comes from a picked center point
    plus a radius, with no GPX track (and so no trail layer)."""
    try:
        job.append_log(f"Working area centered on {lat:.5f}, {lon:.5f} (radius {options.buffer_km}km)")
        track = load_point_area(lat, lon, buffer_km=options.buffer_km)
        job.fetched = fetch_stage(track, options, on_log=job.append_log)
        job.state = "ready"
    except Exception as e:  # noqa: BLE001
        job.error = str(e)
        job.state = "error"


def _run_build(job: Job, adjust: AdjustOptions) -> None:
    try:
        assert job.fetched is not None
        resolved = compose_masks(job.fetched, adjust)
        result = mesh_stage(job.fetched, adjust, resolved, job.dir, write_preview=True, on_log=job.append_log)
        output_extensions = (".stl", ".3mf", ".png")
        job.files = sorted(
            f
            for f in os.listdir(result.out_dir)
            if os.path.isfile(os.path.join(result.out_dir, f)) and f.lower().endswith(output_extensions)
        )
        job.preview_file = os.path.basename(result.preview_path) if result.preview_path else None
        job.parts = [
            {
                "name": name,
                "faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "volume_mm3": round(float(mesh.volume), 1),
            }
            for name, mesh in result.parts
        ]
        job.landcover_warning = result.landcover_warning
        job.state = "done"
    except Exception as e:  # noqa: BLE001
        job.error = str(e)
        job.state = "error"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # GPX files are small; guard against accidental huge uploads
    os.makedirs(_JOBS_ROOT, exist_ok=True)

    @app.get("/")
    def index():
        defaults = dataclasses.asdict(FetchOptions())
        adjust_defaults = dataclasses.asdict(AdjustOptions())
        layers = adjust_defaults.pop("layers")
        layer_assignment = adjust_defaults.pop("layer_assignment")
        defaults.update(adjust_defaults)
        return render_template(
            "index.html",
            defaults=defaults,
            default_layers=layers,
            default_layer_assignment=layer_assignment,
            layer_labels=LAYER_LABELS,
            source_class_labels=SOURCE_CLASS_LABELS,
        )

    @app.post("/api/fetch")
    def api_fetch():
        upload = request.files.get("gpx")
        if upload is None or upload.filename == "":
            return jsonify({"error": "No GPX file uploaded"}), 400
        filename = secure_filename(upload.filename)
        if not filename.lower().endswith(".gpx"):
            return jsonify({"error": "File must be a .gpx track"}), 400

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(_JOBS_ROOT, job_id)
        os.makedirs(job_dir, exist_ok=True)
        gpx_path = os.path.join(job_dir, filename)
        upload.save(gpx_path)

        try:
            fetch_options = _fetch_options_from_form(request.form)
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid option value: {e}"}), 400

        job = Job(job_id, job_dir)
        _JOBS[job_id] = job
        thread = threading.Thread(target=_run_fetch, args=(job, gpx_path, fetch_options), daemon=True)
        thread.start()

        return jsonify({"job_id": job_id})

    @app.post("/api/fetch_area")
    def api_fetch_area():
        try:
            lat = float(request.form["lat"])
            lon = float(request.form["lon"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Pick a center point on the map first"}), 400
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return jsonify({"error": f"Center point out of range: {lat}, {lon}"}), 400

        try:
            fetch_options = _fetch_options_from_form(request.form)
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid option value: {e}"}), 400

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(_JOBS_ROOT, job_id)
        os.makedirs(job_dir, exist_ok=True)

        job = Job(job_id, job_dir)
        _JOBS[job_id] = job
        thread = threading.Thread(target=_run_fetch_area, args=(job, lat, lon, fetch_options), daemon=True)
        thread.start()

        return jsonify({"job_id": job_id})

    @app.get("/api/jobs/<job_id>")
    def api_job_status(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job id"}), 404
        since = request.args.get("since", default=0, type=int)
        return jsonify(job.snapshot(since))

    @app.post("/api/jobs/<job_id>/preview")
    def api_job_preview(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job id"}), 404
        if job.fetched is None:
            return jsonify({"error": f"Job isn't ready for preview yet (state={job.state})"}), 400

        try:
            adjust = _adjust_options_from_form(request.form)
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid option value: {e}"}), 400

        resolved = compose_masks(job.fetched, adjust)
        plaque_mask, plaque_warning = build_plaque_mask(job.fetched, adjust)
        preview_path = os.path.join(job.dir, "preview.png")
        render_preview_for(
            job.fetched,
            resolved,
            adjust.layers,
            preview_path,
            plaque_mask=plaque_mask,
            plaque_color_hex=adjust.plaque_color_hex,
        )
        job.mark_file_available("preview.png")
        stats = mask_area_percentages(resolved, job.fetched.footprint_mask, job.fetched.grid)
        return jsonify(
            {
                "preview_file": "preview.png",
                "stats": {k: round(v, 1) for k, v in stats.items()},
                "plaque_warning": plaque_warning,
            }
        )

    @app.post("/api/jobs/<job_id>/build")
    def api_job_build(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job id"}), 404
        if job.fetched is None:
            return jsonify({"error": f"Job isn't ready to build yet (state={job.state})"}), 400

        try:
            adjust = _adjust_options_from_form(request.form)
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid option value: {e}"}), 400

        job.state = "meshing"
        thread = threading.Thread(target=_run_build, args=(job, adjust), daemon=True)
        thread.start()

        return jsonify({"ok": True})

    @app.get("/api/jobs/<job_id>/files/<path:filename>")
    def api_job_file(job_id: str, filename: str):
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job id"}), 404
        safe_name = secure_filename(filename)
        if safe_name not in job.files:
            return jsonify({"error": "Unknown file for this job"}), 404
        return send_from_directory(job.dir, safe_name, as_attachment=False)

    return app
