"""Local web UI for the drop-off pipeline.

    venv/bin/python scripts/serve.py
    open http://localhost:8756

Same work `caption_dropoff.py` does, driven from a browser instead of a
command line, so the two choices that actually change the output -- which
acoustic model, and whether to ask a second one -- are visible and
switchable rather than buried in flags.

stdlib http.server on purpose. This is a single-user tool on localhost; a
web framework would be a dependency, a version to pin and a thing to
upgrade, in exchange for routing that fits in forty lines. Uploads are
raw-body PUTs rather than multipart for the same reason: the browser sends
`fetch(url, {body: file})` and the server writes the bytes.

Not hardened, and not meant to be: it binds loopback only, serves one
directory, and runs one job at a time.
"""
import glob
import http.server
import json
import mimetypes
import os
import socket
import socketserver
import sys
import threading
import traceback
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caption_dropoff import AUDIO_EXTS, DOC_EXTS, GENERATED, find_pairs  # noqa: E402

from captionlm.biasable import extract_bias_terms  # noqa: E402
from captionlm.biased_model import load_biased_model  # noqa: E402
from captionlm.build_eval_set import convert_to_wav  # noqa: E402
from captionlm.cli import result_to_srt  # noqa: E402
from captionlm.config import MODEL_ID_110M, MODEL_ID_1_1B  # noqa: E402
from captionlm.doc_import import extract_text  # noqa: E402
from captionlm.fusion import WHISPER_MODEL_ID, fuse_tokens, second_opinion  # noqa: E402
from captionlm.terms import build_context_graph, load_tokenizer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
DROPOFF = os.path.join(ROOT, "dropoff")
PORT = int(os.environ.get("CAPTIONLM_PORT", "8756"))

# Measured on the read-aloud set, 447 terms over 507 spoken occurrences, and
# timed over 681.5s of audio. The UI shows these next to the toggles because
# the trade is not obvious: the small model is twice as fast and a tenth the
# disk for two points of term recall, and the second opinion costs a third of
# the throughput for two more.
PRESETS = {
    "110m": {"model": MODEL_ID_110M, "label": "Small · 438 MB", "speed": "59x realtime"},
    "1.1b": {"model": MODEL_ID_1_1B, "label": "Large · 4.0 GB", "speed": "30x realtime"},
}
SCORES = {
    ("110m", False): {"recall": "0.9073", "wer": "0.0850", "speed": "59x"},
    ("110m", True): {"recall": "0.9250", "wer": "0.0659", "speed": "12x"},
    ("1.1b", False): {"recall": "0.9231", "wer": "0.0712", "speed": "30x"},
    ("1.1b", True): {"recall": "0.9428", "wer": "0.0560", "speed": "10x"},
}

_lock = threading.Lock()
_models: dict[str, tuple] = {}
state: dict = {"running": False, "preset": "110m", "second_opinion": False, "files": [],
               "unpaired": [], "error": None}


def _get_model(model_id: str):
    """Models are big and slow to load; keep each one for the process."""
    if model_id not in _models:
        _models[model_id] = (load_biased_model(model_id), load_tokenizer(model_id))
    return _models[model_id]


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def scan() -> list[dict]:
    """Every audio file in dropoff/, with the document it pairs with.

    Captions and terms are read back off disk rather than kept only in
    memory, so restarting the server does not turn a finished job into a
    row marked "done" with nothing behind it. The list of spans the second
    model corrected is not recovered -- that is a note about the run that
    just happened, not a property of the output.
    """
    rows = []
    for audio, docs in sorted(find_pairs(DROPOFF)):
        base = os.path.splitext(audio)[0]
        srt = _read(base + ".srt")
        rows.append(
            {
                "name": os.path.basename(audio),
                "docs": [os.path.basename(d) for d in docs],
                "size": os.path.getsize(audio),
                "status": "done" if srt else "ready",
                "stage": "",
                "progress": 100 if srt else 0,
                "terms": [t for t in _read(base + ".terms.txt").split("\n") if t][:40],
                "cues": _cues(srt) if srt else [],
                "changes": [],
            }
        )
    return rows


def strays(rows: list[dict]) -> list[str]:
    """Files in the drop-off attached to no recording.

    Two ways to get one. A document dropped without a recording of the same
    name: `find_pairs` is keyed on audio, which is right for captioning -- a
    lone document is nothing to transcribe -- but the UI builds its file list
    from it, so the upload would otherwise land on disk and appear nowhere.
    And captions left behind when a recording is removed outside this UI,
    which are excluded from pairing and so would be invisible forever.

    `.converted.wav` is swept rather than listed. It is a decoding cache, it
    is the size of the recording, and it is rebuilt on demand. The `.srt` is
    the thing the user came for, so it is listed and never deleted for them.
    """
    attached = {d for r in rows for d in r["docs"]} | {r["name"] for r in rows}
    out = []
    for path in glob.glob(os.path.join(DROPOFF, "*")):
        name = os.path.basename(path)
        base = name[: -len(".converted.wav")] if name.endswith(".converted.wav") else None
        if base is not None:
            if not any(r["name"].startswith(base + ".") for r in rows):
                os.remove(path)
            continue
        if name == "README.md" or name in attached:
            continue
        stem = name[: -len(".terms.txt")] if name.endswith(".terms.txt") else os.path.splitext(name)[0]
        if name.endswith((".terms.txt", ".srt")):
            if not any(r["name"].startswith(stem + ".") for r in rows):
                out.append(name)
        elif os.path.splitext(name)[1].lower() in DOC_EXTS:
            out.append(name)
    return sorted(out)


def _set(row: dict, **kw) -> None:
    with _lock:
        row.update(kw)


def _cues(srt: str) -> list[dict]:
    out = []
    for block in srt.strip().split("\n\n"):
        lines = block.split("\n")
        if len(lines) >= 3:
            out.append({"time": lines[1].split(" --> ")[0][:-4], "text": " ".join(lines[2:])})
    return out


def run_job(preset: str, want_second: bool) -> None:
    """Transcribe every pairing in the drop-off directory, in order."""
    try:
        model_id = PRESETS[preset]["model"]
        model, tokenizer = _get_model(model_id)
        blank_idx = len(model.vocabulary)

        for row in state["files"]:
            audio = os.path.join(DROPOFF, row["name"])
            base = os.path.splitext(audio)[0]

            terms: list[str] = []
            docs = row.get("docs") or []
            if docs:
                _set(row, status="working",
                     stage="reading document" if len(docs) == 1 else f"reading {len(docs)} documents",
                     progress=0)
                text = "\n\n".join(extract_text(os.path.join(DROPOFF, d)) for d in docs)
                terms = extract_bias_terms(text)
                with open(base + ".terms.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(terms) + "\n")
                model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
            else:
                model.context_graph = None
            _set(row, terms=terms[:40], status="working")

            wav = audio
            if not audio.lower().endswith(".wav"):
                _set(row, stage="converting audio")
                wav = base + ".converted.wav"
                if not os.path.isfile(wav):
                    convert_to_wav(audio, wav)

            _set(row, stage="transcribing", progress=0)
            result = model.transcribe(
                wav,
                chunk_duration=120.0,
                chunk_callback=lambda done, total, r=row: _set(
                    r, progress=round(100 * done / total) if total else 0
                ),
            )
            before = result.text

            changes: list[dict] = []
            if want_second:
                _set(row, stage="asking the second model", progress=100)
                fused = fuse_tokens(result.tokens, second_opinion(wav), terms)
                from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences

                after = sentences_to_result(tokens_to_sentences(fused))
                changes = _diff(before, after.text)
                result = after

            _set(row, stage="writing captions")
            srt = result_to_srt(result)
            with open(base + ".srt", "w", encoding="utf-8") as f:
                f.write(srt)
            _set(row, status="done", stage="", progress=100, cues=_cues(srt), changes=changes)
    except Exception:
        with _lock:
            state["error"] = traceback.format_exc(limit=3)
            # Whatever row we died on is not working any more. Leaving it
            # marked "working" leaves its progress bar moving for the rest of
            # the session, under an error banner, with nothing behind it.
            for row in state["files"]:
                if row["status"] == "working":
                    row.update(status="failed", stage="", progress=0)
    finally:
        with _lock:
            state["running"] = False


def _diff(before: str, after: str) -> list[dict]:
    """What the second opinion actually changed, so the UI can show it rather
    than assert that something happened."""
    import kaldialign

    a = before.lower().replace(",", "").replace(".", "").split()
    b = after.lower().replace(",", "").replace(".", "").split()
    out, run_a, run_b = [], [], []
    for x, y in kaldialign.align(a, b, "*"):
        if x == y:
            if run_a or run_b:
                out.append({"from": " ".join(run_a), "to": " ".join(run_b)})
                run_a, run_b = [], []
        else:
            if x != "*":
                run_a.append(x)
            if y != "*":
                run_b.append(y)
    if run_a or run_b:
        out.append({"from": " ".join(run_a), "to": " ".join(run_b)})
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")

        if url.path == "/api/state":
            with _lock:
                if not state["running"]:
                    # Rows are kept rather than rebuilt, because they carry the
                    # last run's cues and corrections. Their document list is
                    # not a property of that run though -- dropping a document
                    # next to a recording already on the list has to show up.
                    fresh = scan()
                    known = {r["name"]: r for r in state["files"]}
                    for row in fresh:
                        if row["name"] in known:
                            known[row["name"]]["docs"] = row["docs"]
                        else:
                            state["files"].append(row)
                    present = {r["name"] for r in fresh}
                    state["files"] = [r for r in state["files"] if r["name"] in present]
                    state["unpaired"] = strays(state["files"])
                return self._json(
                    {**state, "scores": SCORES[(state["preset"], state["second_opinion"])],
                     "presets": PRESETS}
                )

        if url.path == "/api/srt":
            name = os.path.basename(query.get("name", [""])[0])
            path = os.path.join(DROPOFF, os.path.splitext(name)[0] + ".srt")
            if not os.path.isfile(path):
                return self._json({"error": "not captioned yet"}, 404)
            with open(path, "rb") as f:
                return self._send(200, f.read(), "text/plain; charset=utf-8")

        return self._json({"error": "no such endpoint"}, 404)

    def do_PUT(self):
        url = urllib.parse.urlparse(self.path)
        if url.path != "/api/upload":
            return self._json({"error": "no such endpoint"}, 404)

        name = os.path.basename(urllib.parse.parse_qs(url.query).get("name", [""])[0])
        if not name or os.path.splitext(name)[1].lower() not in AUDIO_EXTS | DOC_EXTS:
            return self._json({"error": f"{name!r} is not an audio or document file"}, 400)

        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        os.makedirs(DROPOFF, exist_ok=True)
        with open(os.path.join(DROPOFF, name), "wb") as f:
            f.write(data)
        return self._json({"ok": True, "name": name})

    def do_DELETE(self):
        """Remove one dropped file, and anything a run wrote next to it.

        Leaving the .srt/.terms.txt/.converted.wav behind means re-dropping a
        recording of the same name shows the previous run's captions.
        """
        url = urllib.parse.urlparse(self.path)
        if url.path != "/api/file":
            return self._json({"error": "no such endpoint"}, 404)

        name = os.path.basename(urllib.parse.parse_qs(url.query).get("name", [""])[0])
        if not name or os.path.splitext(name)[1].lower() not in AUDIO_EXTS | DOC_EXTS | {".srt"}:
            return self._json({"error": f"{name!r} is not an audio or document file"}, 400)

        with _lock:
            if state["running"]:
                return self._json({"error": "a job is running; wait for it to finish"}, 409)

        path = os.path.join(DROPOFF, name)
        if not os.path.isfile(path):
            return self._json({"error": f"{name!r} is not in the drop-off"}, 404)

        removed = []
        base = os.path.splitext(path)[0]
        for victim in [path] + [base + suffix for suffix in GENERATED]:
            if os.path.isfile(victim):
                os.remove(victim)
                removed.append(os.path.basename(victim))
        with _lock:
            state["files"] = [r for r in state["files"] if r["name"] != name]
        return self._json({"ok": True, "removed": removed})

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        payload = json.loads(body or b"{}")

        if url.path == "/api/settings":
            with _lock:
                if payload.get("preset") in PRESETS:
                    state["preset"] = payload["preset"]
                state["second_opinion"] = bool(payload.get("second_opinion", state["second_opinion"]))
            return self._json({"ok": True})

        if url.path == "/api/run":
            with _lock:
                if state["running"]:
                    return self._json({"error": "a job is already running"}, 409)
                state["error"] = None
                state["files"] = scan()
                if not state["files"]:
                    return self._json({"error": "nothing in dropoff/ to caption"}, 400)
                state["running"] = True
                preset, want_second = state["preset"], state["second_opinion"]
            threading.Thread(target=run_job, args=(preset, want_second), daemon=True).start()
            return self._json({"ok": True})

        return self._json({"error": "no such endpoint"}, 404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Server6(Server):
    address_family = socket.AF_INET6


def main():
    mimetypes.init()
    os.makedirs(DROPOFF, exist_ok=True)
    state["files"] = scan()
    print(f"captionlm  http://localhost:{PORT}   (dropoff/ = {DROPOFF})")

    # Listen on both loopback families. "localhost" resolves to ::1 first in
    # Chrome and to 127.0.0.1 first in curl, so binding only one of them makes
    # the URL work in some clients and fail in others.
    servers = [Server(("127.0.0.1", PORT), Handler)]
    try:
        servers.append(Server6(("::1", PORT), Handler))
    except OSError:
        pass  # no IPv6 loopback on this host; 127.0.0.1 is enough
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
