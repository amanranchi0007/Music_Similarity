"""
Beat / downbeat tracking for DropItRight.

Baseline (Music-Plagiarism-Detection) used a Beat-Transformer checkpoint (215MB,
Western-pop-tuned) feeding madmom DBN decoders. We replace that with compIAM's
TCNTracker (Carnatic-tuned, more robust across the tempo/genre range seen in
regional film music), with a plain madmom fallback that needs no external
checkpoint (madmom ships its own pretrained RNNBeatProcessor/DBN models).

Output shape is kept compatible with what the baseline's downstream code expects:
    beat_times      : 1D array of beat timestamps (seconds)
    downbeat_start  : float, timestamp of the first downbeat
    rhythm          : int, beats per bar (3 or 4)
    bpm             : int, estimated tempo
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

MIN_BPM = 55
MAX_BPM = 230
BEATS_PER_BAR_CANDIDATES = [3, 4, 5, 7, 8]


class BeatTrackingResult:
    def __init__(self, beat_times, downbeat_start, rhythm, bpm, source):
        self.beat_times = beat_times
        self.downbeat_start = downbeat_start
        self.rhythm = rhythm
        self.bpm = bpm
        self.source = source  # "tcn-carnatic" or "madmom-fallback"

    def as_tuple(self):
        return self.beat_times, self.downbeat_start, self.rhythm, self.bpm


def _tcn_carnatic_track(audio_path, sr=44100, min_confidence_beats=8):
    """Try compIAM's TCN Carnatic beat/downbeat tracker."""
    import compiam

    tracker = compiam.load_model("rhythm:tcn-carnatic")
    prediction = tracker.predict(
        audio_path,
        sr=sr,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM,
        beats_per_bar=BEATS_PER_BAR_CANDIDATES,
    )
    # prediction: 2D array/list of [beat_time, beat_position_in_bar]
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape[0] < min_confidence_beats:
        raise RuntimeError(
            f"TCN Carnatic tracker returned too few beats ({prediction.shape[0]}) "
            "-- treating as low confidence."
        )

    beat_times = prediction[:, 0]
    beat_positions = prediction[:, 1].astype(int)

    rhythm = int(np.round(beat_positions.max())) if beat_positions.max() > 0 else 4
    downbeat_mask = beat_positions == 1
    if not downbeat_mask.any():
        raise RuntimeError("TCN Carnatic tracker found no downbeats.")
    downbeat_start = float(beat_times[downbeat_mask][0])

    intervals = np.diff(beat_times)
    if len(intervals) == 0:
        raise RuntimeError("TCN Carnatic tracker beats degenerate to a single point.")
    bpm = int(round(60.0 / np.median(intervals)))

    return BeatTrackingResult(beat_times, downbeat_start, rhythm, bpm, "tcn-carnatic")


def _madmom_fallback_track(audio_path):
    """Plain madmom beat/downbeat tracking. Uses madmom's own bundled
    pretrained RNN models -- no external checkpoint needed, unlike the
    baseline's Beat-Transformer path."""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import (
        RNNDownBeatProcessor,
        DBNDownBeatTrackingProcessor,
    )

    beat_act = RNNBeatProcessor()(audio_path)
    beat_tracker = DBNBeatTrackingProcessor(
        min_bpm=MIN_BPM, max_bpm=MAX_BPM, fps=100
    )
    beat_times = beat_tracker(beat_act)

    downbeat_act = RNNDownBeatProcessor()(audio_path)
    downbeat_tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=MIN_BPM, max_bpm=MAX_BPM, fps=100
    )
    downbeat_pred = downbeat_tracker(downbeat_act)
    downbeats = downbeat_pred[downbeat_pred[:, 1] == 1][:, 0]

    if len(beat_times) < 2 or len(downbeats) == 0:
        raise RuntimeError("madmom fallback failed to find a stable beat grid.")

    downbeat_start = float(downbeats[0])
    intervals = np.diff(downbeats)
    rhythm = 4
    if len(intervals) > 0:
        beat_interval = np.median(np.diff(beat_times))
        rhythm_est = int(round(np.median(intervals) / beat_interval))
        rhythm = rhythm_est if rhythm_est in (3, 4) else 4

    bpm = int(round(60.0 / np.median(np.diff(beat_times))))

    return BeatTrackingResult(beat_times, downbeat_start, rhythm, bpm, "madmom-fallback")


def track_beats(audio_path, sr=44100):
    """Beat-track an audio file, preferring the Carnatic-tuned TCN tracker
    and falling back to madmom if TCN fails or reports low confidence."""
    try:
        result = _tcn_carnatic_track(audio_path, sr=sr)
        logger.info("Beat tracking: TCN Carnatic succeeded for %s", audio_path)
        return result
    except Exception as exc:
        logger.warning(
            "Beat tracking: TCN Carnatic failed (%s), falling back to madmom", exc
        )
        return _madmom_fallback_track(audio_path)
