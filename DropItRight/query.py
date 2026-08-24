"""
Query pipeline (plan.md path B: QUERY).

Runs process_song() on a user's uploaded song, does Stage-1 global ANN search
over the reference DB for top-K candidate songs, then Stage-2 segment-level
matching against just those K songs, and prints a Turnitin-style report.

Usage:
    python query.py --audio /path/to/song.wav --db-dir ./reference_db --top-k 5
"""

import argparse
import json
import logging
import jsonpickle

from process_song import process_song
from reference_db import ReferenceDB
from fusion import best_segment_matches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_query(audio_path, db_dir, top_k=5, device="cpu", symbolic_pianoroll=False,
              save_profile=None):
    db = ReferenceDB(db_dir)

    query_info = process_song(
        audio_path, device=device, include_symbolic_pianoroll=symbolic_pianoroll
    )

    if save_profile:
        # Lets visualize.py (or a re-run of this query) reuse the expensive
        # extraction output (MERT/CAE/ASR) without recomputing it.
        with open(save_profile, "w", encoding="utf-8") as f:
            f.write(jsonpickle.encode(query_info, unpicklable=False))
        logger.info("Saved query profile to %s", save_profile)

    if query_info.global_mert_embedding is None:
        raise RuntimeError(
            "Query song has no global_mert_embedding -- MERT extraction must "
            "have failed; check logs above."
        )

    # Stage 1: global candidate shortlist (raga is a soft re-rank nudge, not
    # a hard filter -- see reference_db.search_global's docstring caveat)
    candidates = db.search_global(
        query_info.global_mert_embedding, top_k=top_k, query_raga=query_info.raga
    )
    if not candidates:
        return {"query": audio_path, "matches": [], "message": "No reference songs indexed yet."}

    # Stage 2: segment-level fine matching, only against the shortlist
    candidate_ids = [c["song_id"] for c in candidates]
    profiles = db.load_profiles(candidate_ids)

    report = []
    for candidate in candidates:
        ref_info = profiles[candidate["song_id"]]
        ref_segments = ref_info.get("segments") or []
        query_segments = query_info.segments or []

        segment_matches = best_segment_matches(query_segments, ref_segments, top_n=10)

        report.append({
            "song_id": candidate["song_id"],
            "title": candidate["meta"].get("title"),
            "global_score": candidate["score"],
            "raga": candidate["meta"].get("raga"),
            "segment_matches": segment_matches,
        })

    report.sort(key=lambda r: r["global_score"], reverse=True)
    return {"query": audio_path, "matches": report, "message": "success"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--db-dir", default="reference_db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--symbolic-pianoroll", action="store_true")
    parser.add_argument("--save-profile", default=None,
                         help="Path to save the computed query Music_info profile "
                              "(jsonpickle JSON) for reuse by visualize.py")
    args = parser.parse_args()

    result = run_query(
        args.audio, args.db_dir, top_k=args.top_k, device=args.device,
        symbolic_pianoroll=args.symbolic_pianoroll, save_profile=args.save_profile,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
