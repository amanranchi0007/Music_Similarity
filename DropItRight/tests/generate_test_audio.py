"""
Generates small SYNTHETIC test audio for smoke-testing the DropItRight pipeline
end-to-end (beat tracking -> segmentation -> feature extraction -> indexing ->
query -> segment report).

These are NOT real songs -- no copyrighted or third-party audio is bundled.
They're sine-tone "melodies" over a click track, just structured enough
(regular tempo, bar-aligned notes, one deliberately shared segment between two
files) to exercise the pipeline's plumbing and let you sanity-check that
build_index.py / query.py run end-to-end and that the shared segment shows up
as a high-scoring match in the report.

Do NOT use these to judge match-quality/accuracy of the real embedding models
(MERT/CAE/melodysim) -- sine tones are out-of-distribution for all of them.
Swap in real Indian regional-music audio for actual quality testing.

Usage:
    python tests/generate_test_audio.py
Writes into tests/audio/:
    reference_songs/song_a.wav
    reference_songs/song_b.wav
    query_song.wav   (shares a 4-bar segment with song_a's bars 3-6)
"""

import os
import wave
import numpy as np

SR = 44100
TEMPO_BPM = 100
BEAT_SEC = 60.0 / TEMPO_BPM
BEATS_PER_BAR = 4
BAR_SEC = BEAT_SEC * BEATS_PER_BAR

OUT_DIR = os.path.join(os.path.dirname(__file__), "audio")

# Simple scale in semitone offsets from a 220Hz tonic, loosely evoking a
# raga-like set of scale degrees (Sa Re Ga Ma Pa Dha Ni Sa').
SCALE_SEMITONES = [0, 2, 4, 5, 7, 9, 11, 12]
TONIC_HZ = 220.0


def semitone_to_hz(offset, tonic=TONIC_HZ):
    return tonic * (2 ** (offset / 12.0))


def synth_note(freq, duration_sec, sr=SR, amp=0.3):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # a few harmonics so it's not a bare sine (slightly more voice/instrument-like)
    wave_signal = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    )
    envelope = np.ones_like(t)
    fade = int(sr * min(0.02, duration_sec / 4))
    if fade > 0:
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
    return amp * wave_signal * envelope / np.max(np.abs(wave_signal) + 1e-8)


def synth_click(sr=SR, duration_sec=0.03, amp=0.5):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    click = amp * np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 80)
    return click


def render_melody(bar_patterns, sr=SR):
    """bar_patterns: list of bars, each bar a list of (scale_degree, n_beats)
    covering exactly BEATS_PER_BAR beats. Renders melody + a click on every
    beat so beat trackers have a clean periodic onset to lock onto."""
    total_beats = sum(n for bar in bar_patterns for _, n in bar)
    total_sec = total_beats * BEAT_SEC
    audio = np.zeros(int(sr * total_sec) + sr)  # pad 1s tail

    t_cursor = 0.0
    for bar in bar_patterns:
        for degree, n_beats in bar:
            note_dur = n_beats * BEAT_SEC
            freq = semitone_to_hz(SCALE_SEMITONES[degree % len(SCALE_SEMITONES)])
            note = synth_note(freq, note_dur, sr=sr)
            start_idx = int(t_cursor * sr)
            audio[start_idx:start_idx + len(note)] += note
            t_cursor += note_dur

    # click track, one per beat
    n_beats_total = int(round(t_cursor / BEAT_SEC))
    click = synth_click(sr=sr)
    for b in range(n_beats_total):
        start_idx = int(b * BEAT_SEC * sr)
        audio[start_idx:start_idx + len(click)] += click

    audio = audio[: int(t_cursor * sr)]
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    return audio


def write_wav(path, audio, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    print(f"wrote {path} ({len(audio) / sr:.1f}s)")


def make_bar(degrees):
    """One bar = 4 quarter-note beats, one scale degree per beat."""
    return [(d, 1) for d in degrees]


def main():
    # --- song_a: 8 bars, melody A ---
    melody_a_bars = [
        make_bar([0, 1, 2, 3]),
        make_bar([4, 3, 2, 1]),
        make_bar([0, 2, 4, 5]),  # bar 3 -- this + next 3 bars get "copied" into the query
        make_bar([7, 5, 4, 2]),
        make_bar([0, 1, 2, 0]),
        make_bar([4, 4, 5, 5]),
        make_bar([2, 1, 0, 1]),
        make_bar([0, 0, 0, 0]),
    ]
    song_a = render_melody(melody_a_bars)
    write_wav(os.path.join(OUT_DIR, "reference_songs", "song_a.wav"), song_a)

    # --- song_b: 8 bars, melody B (unrelated to A) ---
    melody_b_bars = [
        make_bar([7, 6, 5, 4]),
        make_bar([3, 4, 5, 6]),
        make_bar([7, 7, 6, 6]),
        make_bar([5, 3, 1, 0]),
        make_bar([2, 3, 4, 5]),
        make_bar([6, 5, 4, 3]),
        make_bar([1, 2, 3, 4]),
        make_bar([0, 0, 0, 0]),
    ]
    song_b = render_melody(melody_b_bars)
    write_wav(os.path.join(OUT_DIR, "reference_songs", "song_b.wav"), song_b)

    # --- query_song: unique intro (from melody B's vocabulary) + bars 3-6 of
    # song_a copied verbatim (the "plagiarized" segment) + unique outro ---
    query_bars = (
        [make_bar([7, 5, 3, 1]), make_bar([2, 4, 6, 5])]
        + melody_a_bars[2:6]  # the copied segment
        + [make_bar([1, 3, 5, 7]), make_bar([0, 0, 0, 0])]
    )
    query_song = render_melody(query_bars)
    write_wav(os.path.join(OUT_DIR, "query_song.wav"), query_song)


if __name__ == "__main__":
    main()
