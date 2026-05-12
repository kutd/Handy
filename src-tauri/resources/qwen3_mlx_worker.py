#!/usr/bin/env python3
"""Persistent Qwen3-ASR MLX worker for Handy.

The Rust app speaks a tiny JSON-lines protocol over stdin/stdout. Keeping this
process alive lets mlx-qwen3-asr reuse its Session and avoid reloading weights
for every transcription.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import traceback

import numpy as np


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

    stream_state = None

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

            if request.get("cmd") == "stream_start":
                request_id = request.get("id")
                start = time.perf_counter()
                stream_state = session.init_streaming(
                    context=request.get("context") or "",
                    language=request.get("language"),
                    chunk_size_sec=float(request.get("chunk_size_sec") or 2.0),
                    max_context_sec=float(request.get("max_context_sec") or 30.0),
                    finalization_mode=request.get("finalization_mode") or "accuracy",
                    endpointing_mode=request.get("endpointing_mode") or "fixed",
                    unfixed_chunk_num=int(request.get("unfixed_chunk_num") or 2),
                    unfixed_token_num=int(request.get("unfixed_token_num") or 5),
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "text": stream_state.text,
                        "stable_text": stream_state.stable_text,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                continue

            if request.get("cmd") == "stream_feed":
                request_id = request.get("id")
                if stream_state is None:
                    raise RuntimeError("Qwen3 MLX streaming state is not active")

                pcm16 = base64.b64decode(request.get("pcm16_b64") or "")
                audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0

                start = time.perf_counter()
                stream_state = session.feed_audio(audio, stream_state)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "text": stream_state.text,
                        "stable_text": stream_state.stable_text,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                continue

            if request.get("cmd") == "stream_finish":
                request_id = request.get("id")
                if stream_state is None:
                    emit({"id": request_id, "ok": True, "text": "", "stable_text": ""})
                    continue

                start = time.perf_counter()
                stream_state = session.finish_streaming(stream_state)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                text = stream_state.text
                stable_text = stream_state.stable_text
                stream_state = None
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "text": text,
                        "stable_text": stable_text,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                continue

            if request.get("cmd") == "stream_cancel":
                stream_state = None
                emit({"id": request.get("id"), "ok": True, "text": "", "stable_text": ""})
                continue

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
