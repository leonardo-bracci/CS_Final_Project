"""
audio_benchmark.py
------------------
Compares faster-whisper (used in transcriber.py) with the original
openai-whisper on the same audio file: model load time, transcription time,
peak memory and the transcribed text.

Usage (from the project root):
    python scripts/audio_benchmark.py [path/to/audio.wav]

Uses samples/sample.wav by default. Model weights download automatically
on first run. Memory is measured separately for each model with tracemalloc.
"""

import sys
import time
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = PROJECT_ROOT / "samples" / "sample.wav"

MODEL_SIZE = "tiny"  # same size used in transcriber.py, for a fair comparison


def benchmark_faster_whisper(audio_path):
    from faster_whisper import WhisperModel

    print(f"\n=== faster-whisper ({MODEL_SIZE}) ===")
    tracemalloc.start()

    load_start = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    load_time = time.perf_counter() - load_start

    start = time.perf_counter()
    segments, info = model.transcribe(str(audio_path), language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    transcribe_time = time.perf_counter() - start

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    print(f"Model load time: {load_time:.2f}s")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Peak Python memory: {peak_mb:.1f} MB")
    print(f"Text: {text}")

    return {"model": f"faster-whisper ({MODEL_SIZE})", "load_s": round(load_time, 2),
            "transcribe_s": round(transcribe_time, 2), "peak_mb": round(peak_mb, 1), "text": text}


def benchmark_openai_whisper(audio_path):
    import whisper

    print(f"\n=== openai-whisper ({MODEL_SIZE}) ===")
    tracemalloc.start()

    load_start = time.perf_counter()
    model = whisper.load_model(MODEL_SIZE)
    load_time = time.perf_counter() - load_start

    start = time.perf_counter()
    result = model.transcribe(str(audio_path), language="en")
    text = result["text"].strip()
    transcribe_time = time.perf_counter() - start

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    print(f"Model load time: {load_time:.2f}s")
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Peak Python memory: {peak_mb:.1f} MB")
    print(f"Text: {text}")

    return {"model": f"openai-whisper ({MODEL_SIZE})", "load_s": round(load_time, 2),
            "transcribe_s": round(transcribe_time, 2), "peak_mb": round(peak_mb, 1), "text": text}


if __name__ == "__main__":
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    results = [
        benchmark_faster_whisper(audio_path),
        benchmark_openai_whisper(audio_path),
    ]

    print("\n\n=== Summary ===")
    print(f"{'Model':<25} | {'Load (s)':>9} | {'Transcribe (s)':>15} | {'Peak MB':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<25} | {r['load_s']:>9} | {r['transcribe_s']:>15} | {r['peak_mb']:>8}")

    print("\nTexts match:", results[0]["text"] == results[1]["text"])
    print("\n(Note: tracemalloc measures Python-level allocations only - it does not")
    print(" capture memory used inside C/C++ extensions such as PyTorch's tensor")
    print(" backend, so these figures likely understate true peak memory. For a")
    print(" fuller picture, cross-check against Task Manager while this script runs.)")
