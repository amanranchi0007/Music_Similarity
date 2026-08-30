"""
Stage-2 segment-level score fusion (plan.md item 9).

Combines the per-segment signals produced by process_song() into one
final_metric per (query_segment, reference_segment) pair, the same role
compare.py's calculate_metric_optimized plays in the baseline for the
symbolic piano-roll term. This module adds the new embedding-based terms
without touching the existing tensor math in compare.py/compare_utils.py.
"""

import math
from collections import Counter

import numpy as np

DEFAULT_WEIGHTS = {
    "cae": 0.35,
    "melodysim": 0.35,
    "lyrics": 0.15,
    "piano_roll": 0.15,  # baseline symbolic signal, only present if
                          # include_symbolic_pianoroll=True in process_song()
}


def _cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _melodysim_similarity(a, b):
    """MelodySim's embedding space is triplet-margin trained on Euclidean
    distance, not cosine -- "same songs are labelled with 0 [distance]" per
    the original repo (see melodysim_embeddings.py docstring). Map distance
    to a bounded (0, 1] similarity via 1/(1+d): d=0 -> 1.0, larger distance
    decays smoothly toward 0 instead of going unbounded/negative."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    distance = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + distance)


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(cand_tokens, ref_tokens, n):
    cand_ngrams = Counter(_ngrams(cand_tokens, n))
    if not cand_ngrams:
        return 0.0
    ref_ngrams = Counter(_ngrams(ref_tokens, n))
    overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in cand_ngrams.items())
    return overlap / sum(cand_ngrams.values())


def _brevity_penalty(cand_tokens, ref_tokens):
    c_len, r_len = len(cand_tokens), len(ref_tokens)
    if c_len == 0:
        return 0.0
    if c_len >= r_len:
        return 1.0
    return math.exp(1 - r_len / c_len)


def _bleu(cand_tokens, ref_tokens, max_n):
    precisions = [_modified_precision(cand_tokens, ref_tokens, n) for n in range(1, max_n + 1)]
    if all(p == 0 for p in precisions):
        return 0.0
    # additive smoothing for zero n-gram precisions (common on short lyric
    # slices, e.g. a 3s segment) instead of letting one zero collapse the
    # whole geometric mean to 0
    smoothed = [p if p > 0 else 1e-9 for p in precisions]
    geo_mean = math.exp(sum(math.log(p) for p in smoothed) / len(smoothed))
    return geo_mean * _brevity_penalty(cand_tokens, ref_tokens)


def _lyrics_similarity(text_a, text_b, max_n=4):
    """N-gram (BLEU-style) lyric similarity: geometric mean of modified
    n-gram precisions (n=1..max_n) with a brevity penalty, computed in both
    directions and averaged since plagiarism matching has no fixed
    "reference" side (BLEU itself is asymmetric -- candidate vs. reference).
    max_n is capped to the shorter lyric slice's length so short per-segment
    transcripts (a handful of words) don't degenerate to an all-zero score."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if not tokens_a or not tokens_b:
        return 0.0
    n = max(1, min(max_n, len(tokens_a), len(tokens_b)))
    score_ab = _bleu(tokens_a, tokens_b, n)
    score_ba = _bleu(tokens_b, tokens_a, n)
    return float((score_ab + score_ba) / 2)


def segment_pair_breakdown(query_segment, ref_segment, piano_roll_score=None):
    """Raw, per-feature similarity scores for one (query, reference) segment
    pair -- the "why did this match" data a visualization needs. Does NOT
    fuse them; see fuse_scores() for that. Any signal missing on either side
    (e.g. melodysim not wired up yet) is simply absent from the returned dict
    rather than being zeroed out, so callers can tell "not computed" apart
    from "computed and low"."""
    scores = {}

    if query_segment.get("cae_embedding") is not None and ref_segment.get("cae_embedding") is not None:
        scores["cae"] = max(0.0, _cosine(query_segment["cae_embedding"], ref_segment["cae_embedding"]))

    if query_segment.get("melodysim_embedding") is not None and ref_segment.get("melodysim_embedding") is not None:
        scores["melodysim"] = _melodysim_similarity(query_segment["melodysim_embedding"], ref_segment["melodysim_embedding"])

    q_lyrics = query_segment.get("lyrics_slice")
    r_lyrics = ref_segment.get("lyrics_slice")
    if q_lyrics or r_lyrics:
        scores["lyrics"] = _lyrics_similarity(q_lyrics, r_lyrics)

    if piano_roll_score is not None:
        scores["piano_roll"] = max(0.0, min(1.0, piano_roll_score))

    return scores


def fuse_scores(scores, weights=None):
    """Combine a per-feature scores dict (from segment_pair_breakdown) into
    one similarity score in [0, 1], renormalizing weights over whichever
    signals are actually present so missing signals degrade gracefully
    instead of crashing or silently zeroing out the score."""
    if not scores:
        return 0.0
    weights = dict(weights or DEFAULT_WEIGHTS)
    active_weights = {k: weights[k] for k in scores if k in weights}
    total_weight = sum(active_weights.values())
    if total_weight == 0:
        return 0.0
    fused = sum(scores[k] * active_weights[k] for k in scores) / total_weight
    return float(np.clip(fused, 0.0, 1.0))


def segment_pair_score(query_segment, ref_segment, piano_roll_score=None,
                        weights=None):
    """Fuse all available per-segment signals for one (query, reference)
    segment pair into a single similarity score in [0, 1]."""
    scores = segment_pair_breakdown(query_segment, ref_segment, piano_roll_score=piano_roll_score)
    return fuse_scores(scores, weights=weights)


def best_segment_matches(query_segments, ref_segments, top_n=10, weights=None):
    """All-pairs segment match for one (query song, reference song) pair,
    returning the top_n highest-scoring segment pairs -- the per-song input
    to the final Turnitin-style report (plan.md item 10). Each result carries
    both the fused score and the raw per-feature breakdown, so a report/
    visualization can show *why* a segment matched, not just that it did."""
    results = []
    for q_seg in query_segments:
        for r_seg in ref_segments:
            breakdown = segment_pair_breakdown(q_seg, r_seg)
            score = fuse_scores(breakdown, weights=weights)
            if score > 0:
                results.append({
                    "score": score,
                    "breakdown": breakdown,
                    "query_start": q_seg["start"],
                    "query_end": q_seg["end"],
                    "ref_start": r_seg["start"],
                    "ref_end": r_seg["end"],
                })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


def all_segment_pairs(query_segments, ref_segments):
    """All-pairs segment breakdown for one (query, reference) song pair, with
    NO fusion applied -- every feature kept as its own score. Used when you
    want to rank/display each signal independently instead of one combined
    decision number (see compare_pair.py)."""
    results = []
    for q_seg in query_segments:
        for r_seg in ref_segments:
            breakdown = segment_pair_breakdown(q_seg, r_seg)
            if not breakdown:
                continue
            results.append({
                "breakdown": breakdown,
                "query_start": q_seg["start"],
                "query_end": q_seg["end"],
                "ref_start": r_seg["start"],
                "ref_end": r_seg["end"],
            })
    return results


def top_matches_by_feature(pairs, feature, top_n=10, min_score=0.5):
    """Sort all_segment_pairs() output by one specific feature's score
    (descending), keeping only pairs where that feature was actually
    computed AND scored at least min_score. No fusion -- this is "top matches
    according to melodysim alone", "top matches according to lyrics alone",
    etc.

    min_score matters: cae/melodysim/lyrics are all continuous scores, so
    "> 0" (the old default) let essentially-noise matches (e.g. cosine 0.02)
    into a "top match" table, which reads as a real overlap when it isn't --
    that's what was making every song look like it had Carnatic-melody
    overlap even when it didn't. 0.5 is a neutral floor across all four
    scales (cae/melodysim cosine-ish, lyrics BLEU, piano_roll clamp), all
    nominally in [0, 1] -- tune per-feature if one signal runs hot/cold."""
    scored = [
        p for p in pairs
        if feature in p["breakdown"] and p["breakdown"][feature] >= min_score
    ]
    scored.sort(key=lambda p: p["breakdown"][feature], reverse=True)
    return scored[:top_n]


def similarity_matrices(query_segments, ref_segments, weights=None):
    """Full all-pairs similarity matrices for one (query, reference) segment
    scale -- one matrix per active feature plus the fused matrix. This is the
    data visualize.py renders as heatmaps.

    Returns:
        {
          "features": ["cae", "lyrics", ...],       # whichever were actually computed
          "fused": np.ndarray [n_query, n_ref],
          "per_feature": {"cae": np.ndarray [n_query, n_ref], ...},
          "query_times": [(start, end), ...],
          "ref_times": [(start, end), ...],
        }
    """
    n_q, n_r = len(query_segments), len(ref_segments)
    fused = np.zeros((n_q, n_r))
    per_feature = {}

    for i, q_seg in enumerate(query_segments):
        for j, r_seg in enumerate(ref_segments):
            breakdown = segment_pair_breakdown(q_seg, r_seg)
            fused[i, j] = fuse_scores(breakdown, weights=weights)
            for feat, val in breakdown.items():
                if feat not in per_feature:
                    per_feature[feat] = np.zeros((n_q, n_r))
                per_feature[feat][i, j] = val

    return {
        "features": sorted(per_feature.keys()),
        "fused": fused,
        "per_feature": per_feature,
        "query_times": [(s["start"], s["end"]) for s in query_segments],
        "ref_times": [(s["start"], s["end"]) for s in ref_segments],
    }
