"""
Standalone visualizer for DropItRight -- separate from query.py/build_index.py
on purpose, so you can re-render reports from already-computed profiles
without re-running the (slow) MERT/CAE/ASR extraction.

Two modes:

  db     Overview of everything currently indexed: song-to-song global
         similarity matrix (from stored MERT embeddings) + raga/tonic table.

  query  For one query song against the reference DB:
           - Global (whole-song) comparison, one panel per feature (MERT,
             lyrics BLEU, raga match, tonic offset) -- not one combined score.
           - Segment-level comparison, one heatmap card per feature (cae,
             melodysim, lyrics, fused) -- each in its own card with a short
             note on what metric/scale it's measuring.
           - A Turnitin-style match timeline connecting the highest-scoring
             segment pairs across all scales.

Both modes render a single self-contained HTML file (PNGs embedded as
base64) -- easy to scp off a headless GPU server and open locally.

Usage:
    python visualize.py db --db-dir ./reference_db --out report_db.html

    python visualize.py query --audio song.wav --db-dir ./reference_db \\
        --top-k 5 --duration-class 5 --out report_query.html --device cuda

    # reuse an already-computed profile (see query.py --save-profile) instead
    # of recomputing MERT/CAE/ASR:
    python visualize.py query --profile query_profile.json --db-dir ./reference_db \\
        --duration-class 5 --out report_query.html
"""

import argparse
import base64
import io
import logging
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless GPU server -- no display needed
import matplotlib.pyplot as plt

from reference_db import ReferenceDB
from fusion import (
    similarity_matrices, best_segment_matches, _lyrics_similarity,
    _melodysim_similarity, DEFAULT_WEIGHTS,
)


def _cosine_pair_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return max(0.0, float(np.dot(a, b) / denom))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Per-feature display metadata: what to call it, how to describe the metric
# underneath it (so a report reader knows e.g. "lyrics" here means n-gram
# BLEU overlap, not semantic similarity), and a distinct colormap so panels
# stay visually distinguishable when scanned side by side.
FEATURE_META = {
    "cae": {
        "label": "CAE-Carnatic",
        "metric": "cosine similarity of CAE-Carnatic melodic-pattern embeddings",
        "cmap": "cividis",
    },
    "melodysim": {
        "label": "MelodySim",
        "metric": "1 / (1 + Euclidean distance) between MelodySim (Siamese/triplet) embeddings",
        "cmap": "PuBu",
    },
    "lyrics": {
        "label": "Lyrics (BLEU)",
        "metric": "symmetric n-gram (BLEU-style, n<=4) overlap of transcribed lyric text",
        "cmap": "YlOrBr",
    },
    "piano_roll": {
        "label": "Piano-roll (symbolic)",
        "metric": "baseline demucs+AST+Beat-Transformer symbolic DTW/correlation score",
        "cmap": "BuGn",
    },
    "mert_global": {
        "label": "MERT",
        "metric": "cosine similarity of whole-song MERT audio embeddings",
    },
    "lyrics_global": {
        "label": "Lyrics (BLEU)",
        "metric": "symmetric n-gram (BLEU-style, n<=4) overlap of the full-song lyrics transcript",
    },
    "tonic_global": {
        "label": "Tonic offset",
        "metric": "difference in whole-song tonic, in cents (0 = identical tonic); not a similarity score",
    },
    "cae_segment_avg": {
        "label": "CAE-Carnatic (segment avg.)",
        "metric": "mean cosine similarity of CAE-Carnatic embeddings across every (query, ref) segment pair",
    },
    "melodysim_segment_avg": {
        "label": "MelodySim (segment avg.)",
        "metric": "mean 1/(1+distance) MelodySim similarity across every (query, ref) segment pair",
    },
}


# --------------------------------------------------------------- plot utils

def _fig_to_base64(fig, transparent=True):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _style_axes_dark(ax, fig):
    """Slightly translucent dark panel so heatmaps/bars blend with the HTML
    report's glass-card background instead of sitting on a hard white/black
    box."""
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0.086, 0.098, 0.114, 0.55))  # ~#161b22 at ~55% opacity
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def _cosine_sim_matrix(vectors):
    mat = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    mat = mat / norms
    return mat @ mat.T


def _heatmap_fig(matrix, title, xlabel, ylabel, xticklabels=None, yticklabels=None,
                  cmap="viridis", vmin=0.0, vmax=1.0):
    n_rows, n_cols = matrix.shape
    fig_w = max(4, min(0.35 * n_cols + 2, 14))
    fig_h = max(3.5, min(0.35 * n_rows + 2, 14))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels is not None and n_cols <= 40:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
    if yticklabels is not None and n_rows <= 40:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(yticklabels, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#c9d1d9")
    _style_axes_dark(ax, fig)
    fig.tight_layout()
    return fig


def _bar_fig(labels, values, xlabel, title, color="#58a6ff", xlim=(0, 1)):
    fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * len(labels))))
    ax.barh(labels, values, color=color, alpha=0.85)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    _style_axes_dark(ax, fig)
    fig.tight_layout()
    return fig


def _timeline_fig(query_duration, ref_duration, matches, score_threshold=0.5,
                   query_title="query", ref_title="reference"):
    """Turnitin-style alignment ribbon: two horizontal timelines with bands
    connecting matched segment pairs, colored/opacity-scaled by score."""
    fig, ax = plt.subplots(figsize=(12, 4))
    y_query, y_ref = 1.0, 0.0
    ax.hlines(y_query, 0, query_duration, color="#c9d1d9", linewidth=4)
    ax.hlines(y_ref, 0, ref_duration, color="#c9d1d9", linewidth=4)
    ax.text(0, y_query + 0.08, query_title, fontsize=10, fontweight="bold", color="#e6edf3")
    ax.text(0, y_ref - 0.16, ref_title, fontsize=10, fontweight="bold", color="#e6edf3")

    cmap = plt.colormaps.get_cmap("autumn_r")
    shown = [m for m in matches if m["score"] >= score_threshold]
    for m in shown:
        color = cmap(min(max(m["score"], 0.0), 1.0))
        poly_x = [m["query_start"], m["query_end"], m["ref_end"], m["ref_start"]]
        poly_y = [y_query, y_query, y_ref, y_ref]
        ax.fill(poly_x, poly_y, color=color, alpha=0.3, edgecolor=color, linewidth=0.5)

    ax.set_xlim(0, max(query_duration, ref_duration))
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Matched segments (score >= {score_threshold})")
    _style_axes_dark(ax, fig)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- pages

HTML_SHELL = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{
    font-family: -apple-system, sans-serif; margin: 24px; color: #e6edf3;
    background: radial-gradient(circle at 20% -10%, #16233a 0%, #0d1117 45%),
                radial-gradient(circle at 90% 10%, #1a1030 0%, #0d1117 50%),
                #0d1117;
    background-attachment: fixed;
  }}
  h1, h2, h3 {{ color: #e6edf3; }}
  img {{ max-width: 100%; margin: 8px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ border: 1px solid rgba(48,54,61,0.7); padding: 6px 10px; text-align: left; }}
  th {{ background: rgba(22,27,34,0.6); }}
  tr:nth-child(even) {{ background: rgba(22,27,34,0.35); }}
  .score {{ font-variant-numeric: tabular-nums; }}

  /* glass-card panels: translucent + blurred so they read as layered over
     the page's background gradient rather than flat opaque boxes */
  .section {{
    margin-bottom: 32px;
    background: rgba(22, 27, 34, 0.55);
    border: 1px solid rgba(48, 54, 61, 0.7);
    border-radius: 12px;
    padding: 18px 22px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
  }}
  .feature-card {{
    margin: 14px 0;
    background: rgba(13, 17, 23, 0.45);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 10px;
    padding: 12px 16px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
  }}
  .feature-card h3 {{ margin: 0 0 2px 0; }}
  .metric-note {{ color: #8b949e; font-size: 12.5px; margin: 0 0 8px 0; font-style: italic; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
  }}
  .badge-match {{ background: rgba(46, 160, 67, 0.25); color: #56d364; }}
  .badge-nomatch {{ background: rgba(139, 148, 158, 0.2); color: #8b949e; }}
</style></head><body>
{body}
</body></html>
"""


def _feature_card(feat, fig_b64):
    meta = FEATURE_META.get(feat, {"label": feat, "metric": "(no description available)"})
    return (
        f'<div class="feature-card"><h3>{meta["label"]}</h3>'
        f'<p class="metric-note">{meta["metric"]}</p>'
        f'<img src="data:image/png;base64,{fig_b64}"></div>'
    )


# -------------------------------------------------------------- db report

def _pairwise_matrix(song_ids, pair_fn, default=0.0):
    n = len(song_ids)
    mat = np.full((n, n), default, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = pair_fn(song_ids[i], song_ids[j])
            mat[i, j] = default if val is None else val
    return mat


def _avg_segment_feature_matrix(song_ids, profiles, feature_key, sim_fn, max_segments=25):
    """Coarse segment-level summary for the DB overview: mean similarity
    across all (query_segment, ref_segment) pairs between two songs, for one
    feature. Segments per song are capped (max_segments) to keep this O(N^2)
    scan of an O(S*S) inner loop bounded for larger reference sets -- this is
    a summary view, not a substitute for the per-segment heatmaps in the
    per-query report."""
    def pair_fn(sid_a, sid_b):
        segs_a = (profiles[sid_a].get("segments") or [])[:max_segments]
        segs_b = (profiles[sid_b].get("segments") or [])[:max_segments]
        vals = []
        for sa in segs_a:
            va = sa.get(feature_key)
            if va is None:
                continue
            for sb in segs_b:
                vb = sb.get(feature_key)
                if vb is None:
                    continue
                vals.append(sim_fn(va, vb))
        return float(np.mean(vals)) if vals else None

    return _pairwise_matrix(song_ids, pair_fn, default=0.0)


def render_db_report(db_dir, out_path):
    db = ReferenceDB(db_dir)
    song_ids = db.list_song_ids()
    if not song_ids:
        logger.warning("No songs indexed in %s -- nothing to visualize.", db_dir)

    profiles = db.load_profiles(song_ids)
    embeddings, titles, ragas = [], [], []
    for sid in song_ids:
        prof = profiles[sid]
        emb = prof.get("global_mert_embedding")
        if emb is None:
            logger.warning("Skipping %s -- no global_mert_embedding stored.", sid)
            continue
        embeddings.append(emb)
        titles.append(prof.get("title", sid))
        ragas.append(prof.get("raga") or "-")

    body = f"<h1>Reference DB overview</h1><p>{len(song_ids)} songs indexed at <code>{db_dir}</code></p>"

    # ---- Global (whole-song), one feature per panel ----
    body += '<div class="section"><h2>Global (whole-song) comparison</h2><p>Song-to-song similarity, one feature at a time.</p>'

    if embeddings:
        sim = _cosine_sim_matrix(embeddings)
        fig = _heatmap_fig(sim, "MERT (whole-song audio embedding)", "song", "song",
                            xticklabels=titles, yticklabels=titles, cmap="magma")
        body += _feature_card("mert_global", _fig_to_base64(fig))

    lyrics_song_ids = [sid for sid in song_ids if (profiles[sid].get("global_lyrics") or {}).get("text")]
    if len(lyrics_song_ids) >= 2:
        lyrics_titles = [profiles[sid].get("title", sid) for sid in lyrics_song_ids]

        def _lyrics_pair(a, b):
            return _lyrics_similarity(profiles[a]["global_lyrics"]["text"], profiles[b]["global_lyrics"]["text"])

        mat = _pairwise_matrix(lyrics_song_ids, _lyrics_pair)
        np.fill_diagonal(mat, 1.0)
        fig = _heatmap_fig(mat, "Lyrics (full-song transcript)", "song", "song",
                            xticklabels=lyrics_titles, yticklabels=lyrics_titles, cmap="YlOrBr")
        body += _feature_card("lyrics_global", _fig_to_base64(fig))
    else:
        body += '<div class="feature-card"><h3>Lyrics (BLEU)</h3><p class="metric-note">Fewer than 2 songs have a lyrics transcript -- nothing to compare yet.</p></div>'

    tonic_song_ids = [sid for sid in song_ids if profiles[sid].get("tonic_hz")]
    if len(tonic_song_ids) >= 2:
        tonic_titles = [profiles[sid].get("title", sid) for sid in tonic_song_ids]

        def _tonic_pair(a, b):
            return _tonic_cents_diff(profiles[a]["tonic_hz"], profiles[b]["tonic_hz"])

        mat = _pairwise_matrix(tonic_song_ids, _tonic_pair)
        bound = max(1.0, np.abs(mat).max())
        fig = _heatmap_fig(mat, "Tonic offset (cents, query row vs. ref column)", "song", "song",
                            xticklabels=tonic_titles, yticklabels=tonic_titles, cmap="coolwarm",
                            vmin=-bound, vmax=bound)
        body += _feature_card("tonic_global", _fig_to_base64(fig))

    raga_labels = [profiles[sid].get("raga") or "-" for sid in song_ids]
    raga_rows = "".join(
        f"<tr><td>{profiles[sid].get('title', sid)}</td><td>{raga_labels[i]}</td></tr>"
        for i, sid in enumerate(song_ids)
    )
    body += (
        '<div class="feature-card"><h3>Raga (DEEPSRGM)</h3>'
        '<p class="metric-note">Soft signal only -- ~40-raga Carnatic vocabulary, a rough proxy outside Carnatic music.</p>'
        f"<table><tr><th>song</th><th>raga</th></tr>{raga_rows}</table></div>"
    )
    body += "</div>"

    # ---- Segment-level, aggregated per song-pair (one feature per panel) ----
    body += (
        '<div class="section"><h2>Segment-level comparison (aggregated)</h2>'
        "<p>Mean similarity across all (query segment, reference segment) pairs between two songs, "
        "per feature -- a coarse summary. For the actual segment-by-segment heatmaps and match timeline "
        "for one specific query, use <code>visualize.py query</code> instead.</p>"
    )
    if len(song_ids) >= 2:
        cae_mat = _avg_segment_feature_matrix(song_ids, profiles, "cae_embedding", _cosine_pair_safe)
        fig = _heatmap_fig(cae_mat, "CAE-Carnatic (mean cosine over segment pairs)", "song", "song",
                            xticklabels=titles, yticklabels=titles, cmap="cividis")
        body += _feature_card("cae_segment_avg", _fig_to_base64(fig))

        melodysim_mat = _avg_segment_feature_matrix(song_ids, profiles, "melodysim_embedding", _melodysim_similarity)
        fig = _heatmap_fig(melodysim_mat, "MelodySim (mean similarity over segment pairs)", "song", "song",
                            xticklabels=titles, yticklabels=titles, cmap="PuBu")
        body += _feature_card("melodysim_segment_avg", _fig_to_base64(fig))
    else:
        body += "<p><i>Need at least 2 indexed songs to compare.</i></p>"
    body += "</div>"

    rows = "".join(
        f"<tr><td>{sid}</td><td>{profiles[sid].get('title')}</td>"
        f"<td>{profiles[sid].get('raga') or '-'}</td>"
        f"<td>{profiles[sid].get('tonic_hz') or '-'}</td>"
        f"<td>{len(profiles[sid].get('segments') or [])}</td></tr>"
        for sid in song_ids
    )
    body += (
        '<div class="section"><h2>Indexed songs</h2><table>'
        "<tr><th>song_id</th><th>title</th><th>raga</th><th>tonic (Hz)</th><th>#segments</th></tr>"
        f"{rows}</table></div>"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HTML_SHELL.format(title="DropItRight — Reference DB", body=body))
    logger.info("Wrote %s", out_path)


# ----------------------------------------------------------- query report

def _load_query_profile(args):
    if args.profile:
        import jsonpickle
        with open(args.profile, "r", encoding="utf-8") as f:
            return jsonpickle.decode(f.read())
    if args.audio:
        from process_song import process_song
        return process_song(args.audio, device=args.device,
                             include_symbolic_pianoroll=args.symbolic_pianoroll)
    raise ValueError("Pass either --audio or --profile")


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _tonic_cents_diff(query_hz, ref_hz):
    if not query_hz or not ref_hz:
        return None
    return 1200.0 * math.log2(query_hz / ref_hz)


def _render_global_section(query_info, candidates):
    """Whole-song comparison, one panel per feature -- MERT cosine (already
    what Stage-1 search ranks on), lyrics BLEU on the full transcripts, raga
    match, tonic offset in cents. Kept separate from the Stage-1 bar chart's
    ranking role: this section is "how similar on each individual signal",
    not "who got shortlisted"."""
    query_lyrics = _get(query_info, "global_lyrics")
    query_raga = _get(query_info, "raga")
    query_tonic = _get(query_info, "tonic_hz")

    labels = [c["meta"].get("title", c["song_id"]) for c in candidates]

    body = '<div class="section"><h2>Global (whole-song) comparison</h2>'
    body += "<p>Each candidate scored independently per feature -- not fused here, so you can see which signal is actually driving (or disagreeing with) the Stage-1 ranking.</p>"

    # MERT (the signal Stage-1 search itself ranked on)
    mert_scores = [c["score"] for c in candidates]
    fig = _bar_fig(labels, mert_scores, "cosine similarity", "MERT (whole-song audio embedding)")
    body += _feature_card("mert_global", _fig_to_base64(fig))

    # Lyrics BLEU on full transcripts (independent of any segment slicing)
    bleu_scores = []
    have_lyrics = query_lyrics and query_lyrics.get("text")
    if have_lyrics:
        for c in candidates:
            ref_profile_lyrics = c.get("_ref_global_lyrics")
            ref_text = ref_profile_lyrics.get("text") if ref_profile_lyrics else None
            bleu_scores.append(_lyrics_similarity(query_lyrics.get("text"), ref_text) if ref_text else 0.0)
        fig = _bar_fig(labels, bleu_scores, "BLEU-style n-gram score", "Lyrics (full-song transcript)",
                        color="#d29922")
        body += _feature_card("lyrics_global", _fig_to_base64(fig))
    else:
        body += '<div class="feature-card"><h3>Lyrics (full-song transcript)</h3><p class="metric-note">No lyrics transcript on the query song -- ASR may have failed or found no vocals.</p></div>'

    # Raga match + tonic offset -- not naturally a [0,1] bar, shown as a table
    raga_rows = []
    for c in candidates:
        ref_raga = c.get("_ref_raga")
        ref_tonic = c.get("_ref_tonic_hz")
        match = query_raga is not None and ref_raga is not None and query_raga == ref_raga
        badge = (f'<span class="badge badge-match">match</span>' if match
                 else f'<span class="badge badge-nomatch">-</span>')
        cents = _tonic_cents_diff(query_tonic, ref_tonic)
        cents_str = f"{cents:+.0f} cents" if cents is not None else "-"
        raga_rows.append(
            f"<tr><td>{c['meta'].get('title', c['song_id'])}</td>"
            f"<td>{query_raga or '-'} vs {ref_raga or '-'} {badge}</td>"
            f"<td>{query_tonic or '-'} Hz vs {ref_tonic or '-'} Hz ({cents_str})</td></tr>"
        )
    body += (
        '<div class="feature-card"><h3>Raga &amp; tonic</h3>'
        '<p class="metric-note">Soft signals only -- DEEPSRGM\'s ~40-raga Carnatic vocabulary is a rough proxy outside Carnatic music; tonic offset shown in cents (0 = identical tonic).</p>'
        "<table><tr><th>candidate</th><th>raga (query vs ref)</th><th>tonic (query vs ref)</th></tr>"
        f"{''.join(raga_rows)}</table></div>"
    )

    body += "</div>"
    return body


def render_query_report(args):
    db = ReferenceDB(args.db_dir)
    query_info = _load_query_profile(args)
    query_title = _get(query_info, "title")
    query_segments_all = _get(query_info, "segments")
    query_mert = _get(query_info, "global_mert_embedding")

    body = f"<h1>Query report — {query_title}</h1>"

    candidates = []
    if query_mert is not None:
        candidates = db.search_global(query_mert, top_k=args.top_k, query_raga=_get(query_info, "raga"))

    if candidates:
        # Stage 1 ranking chart (kept as its own panel -- this is "who got
        # shortlisted", separate from the per-feature global comparison below)
        labels = [c["meta"].get("title", c["song_id"]) for c in candidates]
        scores = [c["score"] for c in candidates]
        fig = _bar_fig(labels, scores, "global similarity (MERT, cosine + raga boost)",
                        "Stage 1 — global candidates")
        body += f'<div class="section"><h2>Stage 1 — global candidates</h2><img src="data:image/png;base64,{_fig_to_base64(fig)}"></div>'

        # Pull each candidate's global lyrics/raga/tonic once for the
        # per-feature global comparison section below.
        for c in candidates:
            ref_profile = db.load_profile(c["song_id"])
            c["_ref_global_lyrics"] = ref_profile.get("global_lyrics")
            c["_ref_raga"] = ref_profile.get("raga")
            c["_ref_tonic_hz"] = ref_profile.get("tonic_hz")

        body += _render_global_section(query_info, candidates)
    else:
        body += "<p><i>No global candidates found (empty DB, or MERT embedding missing for this query).</i></p>"

    ref_song_id = args.ref_song_id or (candidates[0]["song_id"] if candidates else None)
    if ref_song_id is None:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(HTML_SHELL.format(title="DropItRight — Query report", body=body))
        logger.warning("No reference song to compare segments against; wrote candidate-only report.")
        return

    ref_info = db.load_profile(ref_song_id)
    ref_segments_all = ref_info.get("segments") or []

    def _filter(segments, duration_class):
        return [s for s in segments if str(s.get("duration_class")) == str(duration_class)]

    q_segs = _filter(query_segments_all or [], args.duration_class)
    r_segs = _filter(ref_segments_all, args.duration_class)

    body += (
        f'<div class="section"><h2>Stage 2 — segment-level detail vs. '
        f'{ref_info.get("title", ref_song_id)}</h2>'
        f"<p>duration_class = {args.duration_class} &nbsp;|&nbsp; "
        f"{len(q_segs)} query segments x {len(r_segs)} reference segments &nbsp;|&nbsp; "
        f"each feature below is its own panel -- see the note under each for what metric it plots.</p>"
    )

    if q_segs and r_segs:
        mats = similarity_matrices(q_segs, r_segs, weights=args.weights)
        q_labels = [f"{s['start']:.1f}s" for s in q_segs]
        r_labels = [f"{s['start']:.1f}s" for s in r_segs]

        fig = _heatmap_fig(mats["fused"], "Fused segment similarity", "reference segment start",
                            "query segment start", xticklabels=r_labels, yticklabels=q_labels,
                            cmap="magma")
        body += (
            '<div class="feature-card"><h3>Fused</h3>'
            '<p class="metric-note">Weighted combination of every feature below (weights renormalized over whichever signals are present for each segment pair).</p>'
            f'<img src="data:image/png;base64,{_fig_to_base64(fig)}"></div>'
        )

        for feat in mats["features"]:
            meta = FEATURE_META.get(feat, {"cmap": "cividis"})
            fig = _heatmap_fig(mats["per_feature"][feat], f"{FEATURE_META.get(feat, {}).get('label', feat)} similarity",
                                "reference segment start", "query segment start",
                                xticklabels=r_labels, yticklabels=q_labels, cmap=meta["cmap"])
            body += _feature_card(feat, _fig_to_base64(fig))
    else:
        body += "<p><i>No segments at this duration_class on one or both sides -- try a different --duration-class.</i></p>"

    all_matches = best_segment_matches(query_segments_all or [], ref_segments_all,
                                        top_n=200, weights=args.weights)
    query_duration = max((s["end"] for s in (query_segments_all or [])), default=0.0)
    ref_duration = max((s["end"] for s in ref_segments_all), default=0.0)
    fig = _timeline_fig(query_duration, ref_duration, all_matches,
                         score_threshold=args.timeline_threshold,
                         query_title=query_title, ref_title=ref_info.get("title", ref_song_id))
    body += f'<h3>Match timeline (all segment scales)</h3><img src="data:image/png;base64,{_fig_to_base64(fig)}">'

    def _breakdown_str(breakdown):
        parts = [f"{FEATURE_META.get(k, {}).get('label', k)}={v:.2f}" for k, v in breakdown.items()]
        return ", ".join(parts)

    top_rows = "".join(
        "<tr>"
        f"<td class='score'>{m['score']:.3f}</td>"
        f"<td>{m['query_start']:.2f}–{m['query_end']:.2f}s</td>"
        f"<td>{m['ref_start']:.2f}–{m['ref_end']:.2f}s</td>"
        f"<td>{_breakdown_str(m['breakdown'])}</td>"
        "</tr>"
        for m in all_matches[:20]
    )
    body += (
        "<h3>Top 20 segment matches (all scales, with per-feature breakdown)</h3><table>"
        "<tr><th>fused score</th><th>query range</th><th>reference range</th><th>per-feature</th></tr>"
        f"{top_rows}</table></div>"
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(HTML_SHELL.format(title="DropItRight — Query report", body=body))
    logger.info("Wrote %s", args.out)


# ------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    db_parser = sub.add_parser("db", help="Overview of the reference DB")
    db_parser.add_argument("--db-dir", required=True)
    db_parser.add_argument("--out", default="report_db.html")

    q_parser = sub.add_parser("query", help="Report for one query song vs. the reference DB")
    q_parser.add_argument("--audio", default=None, help="Query audio to process (recomputes features)")
    q_parser.add_argument("--profile", default=None, help="Precomputed query profile JSON (see query.py --save-profile)")
    q_parser.add_argument("--db-dir", required=True)
    q_parser.add_argument("--ref-song-id", default=None, help="Force comparison against this song_id (default: top Stage-1 candidate)")
    q_parser.add_argument("--top-k", type=int, default=5)
    q_parser.add_argument("--duration-class", default="5", help="Which segment scale to show as heatmaps: 3, 5, 7, or whole")
    q_parser.add_argument("--timeline-threshold", type=float, default=0.5)
    q_parser.add_argument("--device", default="cpu")
    q_parser.add_argument("--symbolic-pianoroll", action="store_true")
    q_parser.add_argument("--out", default="report_query.html")
    q_parser.set_defaults(weights=DEFAULT_WEIGHTS)

    args = parser.parse_args()

    if args.mode == "db":
        render_db_report(args.db_dir, args.out)
    elif args.mode == "query":
        render_query_report(args)


if __name__ == "__main__":
    main()
