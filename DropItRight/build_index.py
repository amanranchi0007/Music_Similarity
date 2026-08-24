"""
Batch-index reference songs (plan.md path A: INDEXING).

Usage:
    python build_index.py --audio-dir /path/to/reference_songs --db-dir ./reference_db
"""

import os
import argparse
import glob
import logging

from process_song import process_song
from reference_db import ReferenceDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Folder of reference song audio files")
    parser.add_argument("--db-dir", default="reference_db", help="Where to write the reference DB")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--symbolic-pianoroll", action="store_true",
                         help="Also run the baseline demucs+AST piano-roll extraction")
    args = parser.parse_args()

    db = ReferenceDB(args.db_dir)

    audio_files = sorted(
        glob.glob(os.path.join(args.audio_dir, "*.wav"))
        + glob.glob(os.path.join(args.audio_dir, "*.mp3"))
    )
    logger.info("Found %d reference songs to index", len(audio_files))

    for audio_path in audio_files:
        song_id = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            music_info = process_song(
                audio_path,
                title=song_id,
                device=args.device,
                include_symbolic_pianoroll=args.symbolic_pianoroll,
            )
            db.add_song(song_id, music_info)
        except Exception as exc:
            logger.exception("Failed to index %s: %s", audio_path, exc)


if __name__ == "__main__":
    main()
