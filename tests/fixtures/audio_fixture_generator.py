"""
tests/fixtures/audio_fixture_generator.py

Generate synthetic audio fixtures for CI — no real voices, no copyright.
Produces simple sine-wave tones that are valid WAV files the pipeline
can process without error (though the model will return near-random predictions
on synthetic audio — that's expected and is not what we're testing here).
"""
import math, struct, wave
from pathlib import Path


def generate_sine_wav(
    path: str,
    duration_sec: float = 5.0,
    sample_rate:  int   = 16_000,
    frequency_hz: float = 440.0,
    amplitude:    float = 0.3,
) -> None:
    n_samples = int(duration_sec * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            val = int(amplitude * 32767 *
                      math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            wf.writeframes(struct.pack("<h", val))


if __name__ == "__main__":
    out = Path("tests/fixtures")
    out.mkdir(parents=True, exist_ok=True)

    generate_sine_wav(str(out / "audio_5sec.wav"),  duration_sec=5.0)
    generate_sine_wav(str(out / "audio_10sec.wav"), duration_sec=10.0)
    generate_sine_wav(str(out / "audio_0.5sec.wav"), duration_sec=0.5)
    print("[gen] audio_5sec.wav, audio_10sec.wav, audio_0.5sec.wav")
