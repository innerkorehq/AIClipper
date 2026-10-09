"""Threading HTTP sidecar server (stdlib only).

Endpoints (see src/sidecar/client.rs for the Rust shapes):
    GET  /v1/health       -> capability probe
    POST /v1/score_clips   -> Gemma clip scoring
    POST /v1/content_edit  -> Gemma content-edit + shot list
    POST /v1/diarize       -> pyannote diarization of a WAV path
    POST /v1/select_clips  -> Gemma: clips matching client requests
    POST /v1/audio_events  -> laughter / applause / loudness-peak analysis of a WAV path
    POST /v1/shutdown      -> graceful exit (Rust runs this on teardown)
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import asr, audio_events, diarize, gemma

FORCED_MODEL: str | None = None
DIAR_MODEL: str | None = None


def read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw) if raw else {}


def send(handler, code: int, obj) -> None:
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet by default
        pass

    def do_GET(self):
        if self.path == "/v1/health":
            send(self, 200, self._health())
        else:
            send(self, 404, {"ok": False, "message": "not found"})

    def do_POST(self):
        try:
            if self.path == "/v1/score_clips":
                send(self, 200, self._score_clips())
            elif self.path == "/v1/content_edit":
                send(self, 200, self._content_edit())
            elif self.path == "/v1/diarize":
                send(self, 200, self._diarize())
            elif self.path == "/v1/select_clips":
                send(self, 200, self._select_clips())
            elif self.path == "/v1/audio_events":
                send(self, 200, self._audio_events())
            elif self.path == "/v1/transcribe":
                send(self, 200, self._transcribe())
            elif self.path == "/v1/shutdown":
                send(self, 200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                send(self, 404, {"ok": False, "message": "not found"})
        except (json.JSONDecodeError, ValueError) as exc:
            send(self, 400, {"ok": False, "message": str(exc)})
        except Exception as exc:  # model/runtime errors are still JSON
            send(self, 500, {"ok": False, "message": str(exc)})

    def _health(self) -> dict:
        return {
            "ok": gemma.gemma_available(),
            "gemma_loaded": gemma._model is not None,
            "diarization_available": diarize.diarization_available(),
            "models": [FORCED_MODEL] if FORCED_MODEL else [],
            "message": self._health_message(),
        }

    def _health_message(self) -> str:
        if not gemma.gemma_available():
            return "mlx-lm not installed — run: pip3 install -r python_sidecar/requirements.txt"
        return "ready"

    def _model_for(self, body: dict) -> str:
        return str(body.get("model") or FORCED_MODEL or "google/gemma-3-12b-it")

    def _score_clips(self) -> dict:
        body = read_body(self)
        if not gemma.gemma_available():
            raise RuntimeError(self._health_message())
        words = body.get("transcript", {}).get("words", [])
        if not words:
            raise ValueError("empty transcript words")
        candidates, llm_output, full_prompt = gemma.score_clips(
            self._model_for(body),
            words,
            max_clips=int(body.get("max_clips", 32)),
            min_duration=float(body.get("min_duration", 5.0)),
            max_duration=float(body.get("max_duration", 90.0)),
            max_tokens=int(body.get("max_tokens", 4096)),
            extra=body.get("extra_candidates") or None,
        )
        return {"candidates": candidates, "llm_output": llm_output, "prompt": full_prompt}

    def _select_clips(self) -> dict:
        body = read_body(self)
        if not gemma.gemma_available():
            raise RuntimeError(self._health_message())
        words = body.get("transcript", {}).get("words", [])
        requests = body.get("requests") or []
        if not words or not requests:
            raise ValueError("select_clips needs transcript words and requests")
        candidates, best, llm_output, full_prompt = gemma.select_clips(
            self._model_for(body),
            words,
            requests,
            max_tokens=int(body.get("max_tokens", 4096)),
            extra=body.get("extra_candidates") or None,
        )
        return {"candidates": candidates, "best": best, "llm_output": llm_output, "prompt": full_prompt}

    def _audio_events(self) -> dict:
        body = read_body(self)
        wav = body.get("audio")
        if not wav:
            raise ValueError("missing audio path")
        return audio_events.analyze(
            str(wav),
            body.get("model") or None,
            win=float(body.get("win", 3.0)),
            hop=float(body.get("hop", 1.5)),
        )

    def _content_edit(self) -> dict:
        body = read_body(self)
        if not gemma.gemma_available():
            raise RuntimeError(self._health_message())
        candidate = body.get("candidate")
        words = body.get("words", [])
        if not candidate or not words:
            raise ValueError("missing candidate/words")
        scene_cuts = [float(x) for x in body.get("scene_cuts", [])]
        result, llm_output, full_prompt = gemma.content_edit(
            self._model_for(body),
            candidate,
            words,
            scene_cuts,
            max_cut_fraction=float(body.get("max_cut_fraction", 0.25)),
            sample_interval=float(body.get("sample_interval", 0.5)),
            max_tokens=int(body.get("max_tokens", 4096)),
        )
        return {**result, "llm_output": llm_output, "prompt": full_prompt}

    def _diarize(self) -> dict:
        body = read_body(self)
        wav = body.get("audio")
        if not wav:
            raise ValueError("missing `audio` WAV path")
        if not diarize.diarization_available():
            return {"ok": False, "message": "pyannote.audio not installed", "turns": []}
        turns = diarize.diarize(str(wav), DIAR_MODEL)
        return {"ok": True, "turns": turns}

    def _transcribe(self) -> dict:
        body = read_body(self)
        wav = body.get("audio")
        if not wav:
            raise ValueError("missing `audio` WAV path")
        model = body.get("model") or None
        return asr.transcribe(str(wav), model)


def serve(host: str = "127.0.0.1", port: int = 8931, forced_model: str | None = None, diar_model: str | None = None) -> None:
    global FORCED_MODEL, DIAR_MODEL
    if forced_model:
        FORCED_MODEL = forced_model
    DIAR_MODEL = diar_model
    server = ThreadingHTTPServer((host, port), Handler)
    start = time.monotonic()
    print(f"sidecar listening on http://{host}:{port}", flush=True)
    server.timeout = 0.5
    server.serve_forever()
    _ = start
    server.server_close()