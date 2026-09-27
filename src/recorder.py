"""
recorder.py
-----------
Live microphone capture for the spoken questions.

sounddevice is imported inside record_audio(), so the module can be
imported without it installed (e.g. in --demo mode or in tests).

Recording stops when the user presses Enter instead of after a fixed time,
because answers vary in length: a timer would either cut people off or
record silence.
"""

from pathlib import Path


def record_audio(output_path, samplerate=16000, channels=1):
    """Record from the default microphone until Enter is pressed, then save
    to output_path as a WAV file. 16kHz mono matches what Whisper/faster-
    whisper expects internally, avoiding an extra resampling step.

    Returns the Path the recording was saved to.
    """
    import sounddevice as sd
    import soundfile as sf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []

    def callback(indata, frame_count, time_info, status):
        # Called by sounddevice on a background thread for every audio block;
        # we just accumulate blocks here and do the actual file write once
        # recording stops, to keep this callback itself as fast as possible.
        if status:
            print(f"[recorder] stream status: {status}")
        frames.append(indata.copy())

    print("Recording... press Enter to stop.")
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        input()  # blocks until Enter, while the callback keeps collecting frames

    if not frames:
        raise RuntimeError("No audio was captured - check your microphone is connected and not muted.")

    import numpy as np
    audio = np.concatenate(frames, axis=0)
    sf.write(str(output_path), audio, samplerate)
    print(f"Saved recording to {output_path}")
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "samples" / "mic_test.wav"
    record_audio(out_path)
