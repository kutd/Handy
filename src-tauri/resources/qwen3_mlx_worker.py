#!/usr/bin/env python3
"""Persistent Qwen3-ASR MLX worker for Handy.

The Rust app speaks a tiny JSON-lines protocol over stdin/stdout. Keeping this
process alive lets mlx-qwen3-asr reuse its Session and avoid reloading weights
for every transcription.
"""

from __future__ import annotations

import json
import sys
import time
import traceback


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        emit({"ready": False, "error": "usage: qwen3_mlx_worker.py <model-path-or-id>"})
        return 2

    model = sys.argv[1]

    try:
        from mlx_qwen3_asr import Session

        session = Session(model=model)
    except Exception as exc:  # noqa: BLE001 - propagated to Rust as text
        emit({"ready": False, "error": f"failed to load Qwen3 MLX session: {exc}"})
        traceback.print_exc(file=sys.stderr)
        return 1

    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request: dict = {}
        try:
            request = json.loads(line)
            if request.get("cmd") == "shutdown":
                emit({"ok": True})
                return 0

            request_id = request.get("id")
            audio_path = request["audio_path"]
            language = request.get("language")
            context = request.get("context") or ""

            start = time.perf_counter()
            result = session.transcribe(
                audio_path,
                language=language,
                context=context,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "text": result.text,
                    "elapsed_ms": elapsed_ms,
                }
            )
        except Exception as exc:  # noqa: BLE001 - protocol boundary
            traceback.print_exc(file=sys.stderr)
            emit(
                {
                    "id": request.get("id"),
                    "ok": False,
                    "error": str(exc),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
