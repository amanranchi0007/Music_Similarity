"""
Direct 1:1 comparison report for a *known* pair of songs (e.g. a copyright
dispute: "is track X derived from track Y?") -- unlike visualize.py's `query`
mode, this skips Stage 1 entirely (no FAISS candidate search, no reference DB,
no global-MERT ranking chart). You already know which two songs to compare;
this script's only job is to show *where* in each song the overlap is.

Report contents:
  - What was actually extracted for each song: beat-tracking source/tempo,
    raga (Carnatic/DEEPSRGM) + tonic, whether a lyrics transcript exists.
    (piano_roll is a known-missing signal -- see melodysim/piano-roll notes in
    fusion.py; it is deliberately left out of this report until it's wired up.)
  - Segment-level similarity heatmaps, one feature at a time (cae, melodysim,
    lyrics) for every requested duration_class scale (default 3/5/7). No
    fused/combined heatmap -- each signal shown on its own so you can judge
    them independently.
  - Per-feature top-N (default 10) segment match tables, pooled across all
    scales, each feature ranked purely on its own terms -- no fused ranking.
    The lyrics table also shows an example of the actual overlapping n-gram(s)
    behind each BLEU score, not just the number.
  - Per-feature match-timeline plot (top-N matches only, one plot per
    feature) showing *where* in each song the matches sit -- kept readable by
    being both per-feature (not all signals overlaid) and capped at top-N.

Usage:
    python compare_pair.py --query-audio QRY_001_Veera_Raja_Veera.wav \\
        --ref-audio REF_001_Shiva_Stuti.wav --device cuda \\
        --out reports/dispute_QRY_001_vs_REF_001.html

    # reuse precomputed profiles instead of recomputing MERT/CAE/ASR:
    python query.py --audio QRY_001...wav --db-dir ./reference_db --save-profile q.json
    python query.py --audio REF_001...wav --db-dir ./reference_db --save-profile r.json
    python compare_pair.py --query-profile q.json --ref-profile r.json --out report.html
"""

import argparse
import logging
import math

import os

from fusion import all_segment_pairs, top_matches_by_feature, similarity_matrices, DEFAULT_WEIGHTS
from visualize import (
    HTML_SHELL, FEATURE_META, _feature_card, _heatmap_fig, _save_fig,
    _timeline_fig, _get,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Segment-level features to show/rank independently, in display order.
# "fused" is deliberately excluded -- see module docstring. piano_roll is
# left out too -- it's a known-missing signal, never wired end-to-end (no
# per-segment piano-roll score is ever computed by process_song.py/fusion.py).
SEGMENT_FEATURES = ["cae", "melodysim", "lyrics"]


def _load_profile(audio_path, profile_path, device, symbolic_pianoroll, label):
    if profile_path:
        import jsonpickle
        with open(profile_path, "r", encoding="utf-8") as f:
            return jsonpickle.decode(f.read())
    if audio_path:
        from process_song import process_song
        return process_song(audio_path, device=device,
                             include_symbolic_pianoroll=symbolic_pianoroll)
    raise ValueError(f"Pass either --{label}-audio or --{label}-profile")


def _filter(segments, duration_class):
    return [s for s in segments if str(s.get("duration_class")) == str(duration_class)]


def _tonic_cents_diff(query_hz, ref_hz):
    if not query_hz or not ref_hz:
        return None
    return 1200.0 * math.log2(query_hz / ref_hz)


def _segment_lyrics_lookup(segments):
    """(start, end) -> lyrics_slice text, so the lyrics top-N table can show
    the actual matched words behind a BLEU score, not just the number."""
    return {(s["start"], s["end"]): (s.get("lyrics_slice") or "") for s in segments}


def _lyrics_ngram_example(text_a, text_b, max_n=4, top_k=3):
    """Show the actual overlapping n-gram(s) driving a lyrics BLEU score --
    the longest common n-gram found (falling back to shorter n), so "why did
    this segment pair score high on lyrics" has a concrete answer instead of
    just a number. Mirrors fusion._lyrics_similarity's tokenization."""
    if not text_a or not text_b:
        return "(no lyrics on one or both sides)"
    tokens_a, tokens_b = text_a.lower().split(), text_b.lower().split()
    if not tokens_a or not tokens_b:
        return "(no lyrics on one or both sides)"
    top_n = max(1, min(max_n, len(tokens_a), len(tokens_b)))
    for n in range(top_n, 0, -1):
        grams_a = {" ".join(tokens_a[i:i + n]) for i in range(len(tokens_a) - n + 1)}
        grams_b = {" ".join(tokens_b[i:i + n]) for i in range(len(tokens_b) - n + 1)}
        common = grams_a & grams_b
        if common:
            examples = sorted(common, key=len, reverse=True)[:top_k]
            return f'"{"; ".join(examples)}" ({n}-gram)'
    return "no overlapping n-grams"


def _render_extraction_summary(query_info, ref_info, query_title, ref_title):
    """"What are we actually extracting" section, up front: beat tracking
    (source + tempo, per song, since each song is tracked independently --
    there's no shared beat grid between two different recordings), raga +
    tonic, and whether a lyrics transcript exists at all."""
    def beats(info):
        return {
            "source": _get(info, "beat_track_source") or "-",
            "bpm": _get(info, "bpm"),
            "rhythm": _get(info, "rhythm") or "-",
            "n_beats": len(_get(info, "beat_times") or []),
            "downbeat_start": _get(info, "downbeat_start"),
        }

    q_beats, r_beats = beats(query_info), beats(ref_info)

    body = '<div class="section"><h2>What was extracted</h2>'

    body += (
        '<div class="feature-card"><h3>Beat tracking</h3>'
        '<p class="metric-note">TCN Carnatic beat/downbeat tracker (compIAM), falls back to madmom if it fails on a clip. '
        'Each song is tracked independently -- there is no shared beat grid between two different recordings, '
        'this is only used to phrase-align that one song\'s own segmentation.</p>'
        "<table><tr><th></th><th>source</th><th>bpm</th><th>rhythm</th><th>#beats detected</th><th>downbeat start (s)</th></tr>"
        f"<tr><td>{query_title}</td><td>{q_beats['source']}</td><td>{q_beats['bpm'] or '-'}</td>"
        f"<td>{q_beats['rhythm']}</td><td>{q_beats['n_beats']}</td><td>{q_beats['downbeat_start'] if q_beats['downbeat_start'] is not None else '-'}</td></tr>"
        f"<tr><td>{ref_title}</td><td>{r_beats['source']}</td><td>{r_beats['bpm'] or '-'}</td>"
        f"<td>{r_beats['rhythm']}</td><td>{r_beats['n_beats']}</td><td>{r_beats['downbeat_start'] if r_beats['downbeat_start'] is not None else '-'}</td></tr>"
        "</table></div>"
    )

    query_raga, ref_raga = _get(query_info, "raga"), _get(ref_info, "raga")
    query_tonic, ref_tonic = _get(query_info, "tonic_hz"), _get(ref_info, "tonic_hz")
    match = query_raga is not None and ref_raga is not None and query_raga == ref_raga
    badge = ('<span class="badge badge-match">match</span>' if match
             else '<span class="badge badge-nomatch">-</span>')
    cents = _tonic_cents_diff(query_tonic, ref_tonic)
    cents_str = f"{cents:+.0f} cents" if cents is not None else "-"
    body += (
        '<div class="feature-card"><h3>Raga (DEEPSRGM, Carnatic) &amp; tonic</h3>'
        '<p class="metric-note">Soft signal only -- ~40-raga Carnatic vocabulary, a rough proxy outside Carnatic music. '
        'Tonic from TonicIndianMultiPitch, offset shown in cents (0 = identical tonic).</p>'
        "<table><tr><th></th><th>raga</th><th>tonic (Hz)</th></tr>"
        f"<tr><td>{query_title}</td><td>{query_raga or '-'}</td><td>{query_tonic or '-'}</td></tr>"
        f"<tr><td>{ref_title}</td><td>{ref_raga or '-'}</td><td>{ref_tonic or '-'}</td></tr>"
        f"<tr><td colspan='3'>raga {badge} &nbsp;|&nbsp; tonic offset: {cents_str}</td></tr>"
        "</table></div>"
    )

    query_lyrics = _get(query_info, "global_lyrics") or {}
    ref_lyrics = _get(ref_info, "global_lyrics") or {}
    body += (
        '<div class="feature-card"><h3>Lyrics transcript</h3>'
        "<table><tr><th></th><th>transcript found</th><th>#words (approx.)</th></tr>"
        f"<tr><td>{query_title}</td><td>{'yes' if query_lyrics.get('text') else 'no'}</td>"
        f"<td>{len(query_lyrics.get('text', '').split())}</td></tr>"
        f"<tr><td>{ref_title}</td><td>{'yes' if ref_lyrics.get('text') else 'no'}</td>"
        f"<td>{len(ref_lyrics.get('text', '').split())}</td></tr>"
        "</table></div>"
    )

    body += "</div>"
    return body


def render_pair_report(args):
    query_info = _load_profile(args.query_audio, args.query_profile, args.device,
                                args.symbolic_pianoroll, "query")
    ref_info = _load_profile(args.ref_audio, args.ref_profile, args.device,
                              args.symbolic_pianoroll, "ref")

    query_title = _get(query_info, "title") or "query"
    ref_title = _get(ref_info, "title") or "reference"
    query_segments_all = _get(query_info, "segments") or []
    ref_segments_all = _get(ref_info, "segments") or []

    # Every plot is auto-saved (300dpi PNG) next to the HTML report as it's
    # generated -- no manual "download plot" step needed.
    plots_dir = os.path.splitext(args.out)[0] + "_plots"

    body = f"<h1>Dispute report — {query_title} vs. {ref_title}</h1>"
    body += (
        "<p>Direct segment-level comparison of a known pair -- no Stage 1 "
        "candidate search. Every signal is shown separately below "
        "(no fused/combined score) so each can be judged on its own.</p>"
    )

    body += _render_extraction_summary(query_info, ref_info, query_title, ref_title)

    duration_classes = [d.strip() for d in args.duration_classes.split(",") if d.strip()]

    for dclass in duration_classes:
        q_segs = _filter(query_segments_all, dclass)
        r_segs = _filter(ref_segments_all, dclass)

        body += (
            f'<div class="section"><h2>Segment-level detail — scale {dclass}</h2>'
            f"<p>{len(q_segs)} query segments x {len(r_segs)} reference segments"
            f"&nbsp;|&nbsp; each feature below is its own panel, no fused score.</p>"
        )

        if q_segs and r_segs:
            mats = similarity_matrices(q_segs, r_segs, weights=args.weights)
            q_labels = [f"{s['start']:.1f}s" for s in q_segs]
            r_labels = [f"{s['start']:.1f}s" for s in r_segs]

            for feat in SEGMENT_FEATURES:
                if feat not in mats["per_feature"]:
                    continue  # signal not computed for this pair (e.g. melodysim ckpt not set)
                meta = FEATURE_META.get(feat, {"cmap": "cividis"})
                fig = _heatmap_fig(
                    mats["per_feature"][feat],
                    f"{FEATURE_META.get(feat, {}).get('label', feat)} similarity",
                    "reference segment start", "query segment start",
                    xticklabels=r_labels, yticklabels=q_labels, cmap=meta["cmap"],
                )
                fig_path = os.path.join(plots_dir, f"heatmap_{feat}_scale{dclass}.png")
                body += _feature_card(feat, _save_fig(fig, fig_path))
        else:
            body += "<p><i>No segments at this duration_class on one or both sides -- skipping.</i></p>"

        body += "</div>"

    # Per-feature top-N tables, pooled across ALL scales, ranked independently
    # (no fusion) -- "top matches according to melodysim alone", etc.
    all_pairs = all_segment_pairs(query_segments_all, ref_segments_all)
    q_lyrics_lookup = _segment_lyrics_lookup(query_segments_all)
    r_lyrics_lookup = _segment_lyrics_lookup(ref_segments_all)

    body += (
        '<div class="section"><h2>Top segment matches, per feature (all scales)</h2>'
        f"<p>Each feature ranked on its own -- no combined score. Top {args.top_n} shown per feature.</p>"
    )

    top_n = args.top_n
    # cae was the noisy signal (near-zero cosine scores reading as "matches");
    # melodysim/lyrics/piano_roll keep their original >0 sensitivity.
    min_score_by_feature = {
        "cae": args.min_score_cae,
        "melodysim": args.min_score_melodysim,
        "lyrics": args.min_score_lyrics,
    }
    for feat in SEGMENT_FEATURES:
        matches = top_matches_by_feature(all_pairs, feat, top_n=top_n, min_score=min_score_by_feature.get(feat, 0.0))
        label = FEATURE_META.get(feat, {}).get("label", feat)
        if not matches:
            body += f'<div class="feature-card"><h3>{label}</h3><p class="metric-note">No matches (signal not computed for this pair, or all scores were 0).</p></div>'
            continue

        extra_header = "<th>example matched n-gram</th>" if feat == "lyrics" else ""
        rows = ""
        for m in matches:
            extra_cell = ""
            if feat == "lyrics":
                q_text = q_lyrics_lookup.get((m["query_start"], m["query_end"]), "")
                r_text = r_lyrics_lookup.get((m["ref_start"], m["ref_end"]), "")
                extra_cell = f"<td>{_lyrics_ngram_example(q_text, r_text)}</td>"
            rows += (
                "<tr>"
                f"<td class='score'>{m['breakdown'][feat]:.3f}</td>"
                f"<td>{m['query_start']:.2f}–{m['query_end']:.2f}s</td>"
                f"<td>{m['ref_start']:.2f}–{m['ref_end']:.2f}s</td>"
                f"{extra_cell}"
                "</tr>"
            )
        body += (
            f'<div class="feature-card"><h3>{label} — top {len(matches)}</h3>'
            f'<p class="metric-note">{FEATURE_META.get(feat, {}).get("metric", "")}</p>'
            "<table><tr><th>score</th>"
            f"<th>{query_title} time range</th><th>{ref_title} time range</th>{extra_header}</tr>"
            f"{rows}</table></div>"
        )

    body += "</div>"

    # Per-feature match-timeline plots: where in each song do the top-N
    # matches sit. One plot per feature (not all signals overlaid) and capped
    # at top-N, so this stays readable -- unlike the old all-features ribbon.
    query_duration = max((s["end"] for s in query_segments_all), default=0.0)
    ref_duration = max((s["end"] for s in ref_segments_all), default=0.0)

    body += '<div class="section"><h2>Match timeline, per feature (top matches only)</h2>'
    for feat in SEGMENT_FEATURES:
        matches = top_matches_by_feature(all_pairs, feat, top_n=top_n, min_score=min_score_by_feature.get(feat, 0.0))
        label = FEATURE_META.get(feat, {}).get("label", feat)
        if not matches or query_duration == 0.0 or ref_duration == 0.0:
            continue
        timeline_matches = [{**m, "score": m["breakdown"][feat]} for m in matches]
        fig = _timeline_fig(
            query_duration, ref_duration, timeline_matches, score_threshold=0.0,
            query_title=query_title, ref_title=ref_title,
        )
        fig_path = os.path.join(plots_dir, f"timeline_{feat}.png")
        body += _feature_card(feat, _save_fig(fig, fig_path))
    body += "</div>"

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(HTML_SHELL.format(title=f"DropItRight — {query_title} vs {ref_title}", body=body))
    logger.info("Wrote %s", args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query-audio", default=None, help="First (query/allegedly-copying) track")
    parser.add_argument("--query-profile", default=None, help="Precomputed profile JSON for the query track")
    parser.add_argument("--ref-audio", default=None, help="Second (reference/original) track")
    parser.add_argument("--ref-profile", default=None, help="Precomputed profile JSON for the reference track")
    parser.add_argument("--duration-classes", default="3,5,7",
                         help="Comma-separated segment scales to render heatmaps for (default: 3,5,7)")
    parser.add_argument("--top-n", type=int, default=5, help="How many top segment matches to list (and plot on the timeline), per feature")
    parser.add_argument("--min-score-cae", type=float, default=0.5,
                         help="Minimum cae score for a segment pair to count as a 'match' (default: 0.5 -- cae was the noisy one, everything else stays at its original sensitivity)")
    parser.add_argument("--min-score-melodysim", type=float, default=0.0,
                         help="Minimum melodysim score for a segment pair to count as a 'match' (default: 0.0, i.e. original >0 behavior)")
    parser.add_argument("--min-score-lyrics", type=float, default=0.0,
                         help="Minimum lyrics score for a segment pair to count as a 'match' (default: 0.0, i.e. original >0 behavior)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--symbolic-pianoroll", action="store_true")
    parser.add_argument("--out", default="report_dispute.html")
    parser.set_defaults(weights=DEFAULT_WEIGHTS)

    args = parser.parse_args()
    render_pair_report(args)


if __name__ == "__main__":
    main()
