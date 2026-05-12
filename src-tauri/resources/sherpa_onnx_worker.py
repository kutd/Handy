#!/usr/bin/env python3
"""Persistent sherpa-onnx worker for Handy's Korean Zipformer model."""

import base64
import json
import sys
import time
import traceback
import wave
from array import array
from pathlib import Path


SAMPLE_RATE = 16000


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def find_required(model_dir, name):
    path = model_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing required model file: {path}")
    return str(path)


def create_recognizer(model_dir):
    import sherpa_onnx

    encoder = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
    joiner = model_dir / "joiner-epoch-99-avg-1.int8.onnx"
    if not encoder.exists():
        encoder = model_dir / "encoder-epoch-99-avg-1.onnx"
    if not joiner.exists():
        joiner = model_dir / "joiner-epoch-99-avg-1.onnx"

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=find_required(model_dir, "tokens.txt"),
        encoder=str(encoder),
        decoder=find_required(model_dir, "decoder-epoch-99-avg-1.onnx"),
        joiner=str(joiner),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="modified_beam_search",
        provider="cpu",
    )


def pcm16_b64_to_floats(value):
    pcm = base64.b64decode(value or "")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return [sample / 32768.0 for sample in samples]


def read_wav(path):
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError(f"unsupported wav sample width: {width}")

    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()

    if channels > 1:
        samples = array("h", samples[0::channels])

    return sample_rate, [sample / 32768.0 for sample in samples]


def decode_available(recognizer, stream):
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)


def finish_stream(recognizer, stream, sample_rate):
    stream.accept_waveform(sample_rate, [0.0] * int(sample_rate * 0.5))
    stream.input_finished()
    decode_available(recognizer, stream)
    return recognizer.get_result(stream).strip()


def transcribe_audio(recognizer, samples, sample_rate):
    stream = recognizer.create_stream()
    chunk_size = max(1, int(sample_rate * 0.32))
    for start in range(0, len(samples), chunk_size):
        stream.accept_waveform(sample_rate, samples[start : start + chunk_size])
        decode_available(recognizer, stream)
    return finish_stream(recognizer, stream, sample_rate)


def main():
    if len(sys.argv) != 2:
        emit({"ready": False, "error": "usage: sherpa_onnx_worker.py <model-dir>"})
        return 2

    model_dir = Path(sys.argv[1])
    stream = None

    try:
        recognizer = create_recognizer(model_dir)
        emit({"ready": True})
    except Exception as exc:
        emit({"ready": False, "error": str(exc)})
        traceback.print_exc(file=sys.stderr)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            request_id = request.get("id")
            cmd = request.get("cmd")

            if cmd == "shutdown":
                emit({"ok": True})
                return 0

            if cmd == "transcribe":
                start = time.perf_counter()
                sample_rate, samples = read_wav(request.get("audio_path"))
                text = transcribe_audio(recognizer, samples, sample_rate)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "text": text,
                        "stable_text": text,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                continue

            if cmd == "stream_start":
                start = time.perf_counter()
                hotwords = (request.get("hotwords") or "").strip()
                stream = recognizer.create_stream(hotwords) if hotwords else recognizer.create_stream()
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "text": "",
                        "stable_text": "",
                        "elapsed_ms": elapsed_ms,
                    }
                )
                continue

            if cmd == "stream_feed":
                if stream is None:
                    raise RuntimeError("sherpa-onnx streaming state is not active")

                start = time.perf_counter()
                samples = pcm16_b64_to_floats(request.get("pcm16_b64"))
                stream.accept_waveform(SAMPLE_RATE, samples)
                decode_available(recognizer, stream)
                text = recognizer.get_result(stream).strip()
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "text": text,
                        "stable_text": text,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                continue

            if cmd == "stream_cancel":
                stream = None
                emit({"id": request_id, "ok": True, "text": "", "stable_text": ""})
                continue

            raise ValueError(f"unknown command: {cmd}")
        except Exception as exc:
            emit(
                {
                    "id": request.get("id") if "request" in locals() else None,
                    "ok": False,
                    "error": str(exc),
                }
            )
            traceback.print_exc(file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
