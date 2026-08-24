"""
Phrase-aligned segmentation.

Baseline (Music-Plagiarism-Detection) grouped audio into fixed 4-bar windows
(compare_utils.infos_to_startpoint / refine_breakpoints_custom), independent of
where musical phrases actually start/end.

Here we group consecutive bars (from beat_tracking.track_beats) into segments
whose duration approximates each of the target scales (3s/5s/7s + whole song),
but whose boundaries always land on a real downbeat -- never mid-phrase.
"""

import numpy as np

DEFAULT_TARGET_DURATIONS = [3, 5, 7]


class Segment:
    def __init__(self, start, end, duration_class, bar_start_idx, bar_end_idx):
        self.start = float(start)
        self.end = float(end)
        self.duration_class = duration_class  # 3, 5, 7, or "whole"
        self.bar_start_idx = bar_start_idx
        self.bar_end_idx = bar_end_idx

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "duration_class": self.duration_class,
            "bar_start_idx": self.bar_start_idx,
            "bar_end_idx": self.bar_end_idx,
        }


def _bar_boundaries(beat_times, downbeat_start, rhythm):
    """Return the timestamps of every bar line (downbeat) in the song,
    derived from a regular beat grid anchored at downbeat_start."""
    beat_times = np.asarray(beat_times)
    beat_interval = np.median(np.diff(beat_times))
    bar_interval = beat_interval * rhythm

    n_bars = int(np.floor((beat_times[-1] - downbeat_start) / bar_interval)) + 1
    bar_times = downbeat_start + np.arange(n_bars) * bar_interval
    bar_times = bar_times[(bar_times >= beat_times[0]) & (bar_times <= beat_times[-1])]
    return bar_times, bar_interval


def segment_phrases(
    beat_times,
    downbeat_start,
    rhythm,
    target_durations=DEFAULT_TARGET_DURATIONS,
    include_whole_song=True,
):
    """Group bars into phrase-aligned segments near each target duration.

    Returns dict: {duration_class: [Segment, ...]}
    """
    bar_times, bar_interval = _bar_boundaries(beat_times, downbeat_start, rhythm)
    if len(bar_times) < 2:
        raise RuntimeError("Not enough bars to segment -- check beat tracking output.")

    song_end = float(beat_times[-1])
    segments_by_scale = {}

    for target in target_durations:
        n_bars_per_segment = max(1, int(round(target / bar_interval)))
        segments = []
        bar_idx = 0
        while bar_idx < len(bar_times) - 1:
            end_idx = min(bar_idx + n_bars_per_segment, len(bar_times) - 1)
            start_t = bar_times[bar_idx]
            end_t = bar_times[end_idx]
            if end_t > start_t:
                segments.append(
                    Segment(start_t, end_t, duration_class=target,
                             bar_start_idx=bar_idx, bar_end_idx=end_idx)
                )
            bar_idx = end_idx
        segments_by_scale[target] = segments

    if include_whole_song:
        segments_by_scale["whole"] = [
            Segment(bar_times[0], song_end, duration_class="whole",
                     bar_start_idx=0, bar_end_idx=len(bar_times) - 1)
        ]

    return segments_by_scale


def segment_phrases_fixed_window(audio_duration, target_durations=DEFAULT_TARGET_DURATIONS,
                                   include_whole_song=True):
    """Fallback: fixed-window segmentation, used only when beat tracking
    (both TCN and madmom) fails entirely so the pipeline still produces
    *some* segments rather than blocking."""
    segments_by_scale = {}
    for target in target_durations:
        segments = []
        t = 0.0
        while t < audio_duration:
            end_t = min(t + target, audio_duration)
            if end_t - t > 0.5:  # drop degenerate slivers
                segments.append(Segment(t, end_t, duration_class=target,
                                          bar_start_idx=None, bar_end_idx=None))
            t = end_t
        segments_by_scale[target] = segments

    if include_whole_song:
        segments_by_scale["whole"] = [
            Segment(0.0, audio_duration, duration_class="whole",
                     bar_start_idx=None, bar_end_idx=None)
        ]
    return segments_by_scale
