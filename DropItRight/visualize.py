"""
Standalone visualizer for DropItRight -- separate from query.py/build_index.py
on purpose, so you can re-render reports from already-computed profiles
without re-running the (slow) MERT/CAE/ASR extraction.

Two modes:

  db     Overview of everything currently indexed: song-to-song global
         similarity matrix (from stored MERT embeddings) + raga/tonic table.

  query  For one query song against the reference DB: global candidate
         scores, per-feature segment-level similarity heatmaps (cae,
         melodysim, lyrics, fused) for a chosen reference song, and a
         Turnitin-style match timeline connecting the highest-scoring
         segment pairs.

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

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless GPU server -- no display needed
import matplotlib.pyplot as plt

from reference_db import ReferenceDB
from fusion import similarity_matrices, best_segment_matches, DEFAULT_WEIGHTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------- plot utils

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


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
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels is not None and n_cols <= 40:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
    if yticklabels is not None and n_rows <= 40:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(yticklabels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _timeline_fig(query_duration, ref_duration, matches, score_threshold=0.5,
                   query_title="query", ref_title="reference"):
    """Turnitin-style alignment ribbon: two horizontal timelines with bands
    connecting matched segment pairs, colored/opacity-scaled by score."""
    fig, ax = plt.subplots(figsize=(12, 4))
    y_query, y_ref = 1.0, 0.0
    ax.hlines(y_query, 0, query_duration, color="black", linewidth=4)
    ax.hlines(y_ref, 0, ref_duration, color="black", linewidth=4)
    ax.text(0, y_query + 0.08, query_title, fontsize=10, fontweight="bold")
    ax.text(0, y_ref - 0.16, ref_title, fontsize=10, fontweight="bold")

    cmap = plt.colormaps.get_cmap("autumn_r")
    shown = [m for m in matches if m["score"] >= score_threshold]
    for m in shown:
        color = cmap(min(max(m["score"], 0.0), 1.0))
        poly_x = [m["query_start"], m["query_end"], m["ref_end"], m["ref_start"]]
        poly_y = [y_query, y_query, y_ref, y_ref]
        ax.fill(poly_x, poly_y, color=color, alpha=0.35, edgecolor=color, linewidth=0.5)

    ax.set_xlim(0, max(query_duration, ref_duration))
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Matched segments (score >= {score_threshold})")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- pages

HTML_SHELL = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 24px; background: #0d1117; color: #e6edf3; }}
  h1, h2 {{ color: #e6edf3; }}
  img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 6px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; }}
  th {{ background: #161b22; }}
  tr:nth-child(even) {{ background: #161b22; }}
  .section {{ margin-bottom: 40px; }}
  .score {{ font-variant-numeric: tabular-nums; }}
</style></head><body>
{body}
</body></html>
"""


# -------------------------------------------------------------- db report

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

    if embeddings:
        sim = _cosine_sim_matrix(embeddings)
        fig = _heatmap_fig(sim, "Song-to-song global similarity (MERT, cosine)",
                            "song", "song", xticklabels=titles, yticklabels=titles,
                            cmap="magma")
        body += f'<div class="section"><h2>Global similarity matrix</h2><img src="data:image/png;base64,{_fig_to_base64(fig)}"></div>'

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


def render_query_report(args):
    db = ReferenceDB(args.db_dir)
    query_info = _load_query_profile(args)
    query_title = query_info.get("title") if isinstance(query_info, dict) else query_info.title
    query_segments_all = query_info.get("segments") if isinstance(query_info, dict) else query_info.segments
    query_mert = query_info.get("global_mert_embedding") if isinstance(query_info, dict) else query_info.global_mert_embedding

    body = f"<h1>Query report — {query_title}</h1>"

    candidates = []
    if query_mert is not None:
        candidates = db.search_global(query_mert, top_k=args.top_k)

    if candidates:
        fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * len(candidates))))
        labels = [c["meta"].get("title", c["song_id"]) for c in candidates]
        scores = [c["score"] for c in candidates]
        ax.barh(labels, scores, color="#58a6ff")
        ax.set_xlim(0, 1)
        ax.set_xlabel("global similarity (MERT, cosine)")
        ax.invert_yaxis()
        fig.tight_layout()
        body += f'<div class="section"><h2>Stage 1 — global candidates</h2><img src="data:image/png;base64,{_fig_to_base64(fig)}"></div>'
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
        f"{len(q_segs)} query segments x {len(r_segs)} reference segments</p>"
    )

    if q_segs and r_segs:
        mats = similarity_matrices(q_segs, r_segs, weights=args.weights)
        q_labels = [f"{s['start']:.1f}s" for s in q_segs]
        r_labels = [f"{s['start']:.1f}s" for s in r_segs]

        fig = _heatmap_fig(mats["fused"], "Fused segment similarity", "reference segment start",
                            "query segment start", xticklabels=r_labels, yticklabels=q_labels)
        body += f'<h3>Fused</h3><img src="data:image/png;base64,{_fig_to_base64(fig)}">'

        for feat in mats["features"]:
            fig = _heatmap_fig(mats["per_feature"][feat], f"{feat} similarity",
                                "reference segment start", "query segment start",
                                xticklabels=r_labels, yticklabels=q_labels, cmap="cividis")
            body += f'<h3>{feat}</h3><img src="data:image/png;base64,{_fig_to_base64(fig)}">'
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

    top_rows = "".join(
        "<tr>"
        f"<td class='score'>{m['score']:.3f}</td>"
        f"<td>{m['query_start']:.2f}–{m['query_end']:.2f}s</td>"
        f"<td>{m['ref_start']:.2f}–{m['ref_end']:.2f}s</td>"
        f"<td>{', '.join(f'{k}={v:.2f}' for k, v in m['breakdown'].items())}</td>"
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
