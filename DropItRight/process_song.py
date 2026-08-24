"""
process_song(): the single entrypoint both indexing and query paths call
(plan.md, top of pipeline). Produces one extended Music_info profile per song:

  separation -> beat tracking -> phrase segmentation -> global features
  -> per-segment features -> Music_info JSON

Reuses the baseline's demucs separation + AST vocal transcription
(baseline_segment_transcription.py) for the parts that don't need to change,
and adds the new Indian-feature / embedding / segmentation stages around it.
"""

import os
import logging
import soundfile as sf

import beat_tracking
import segmentation
import indian_features
import global_embeddings
import melodysim_embeddings
from music_info import Music_info

logger = logging.getLogger(__name__)


def _render_segment_wav(audio_path, start, end, out_path, sr=44100):
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=max(end - start, 0.05))
    sf.write(out_path, y, sr)
    return out_path


def process_song(audio_path, title=None, tmp_dir="tmp_segments", device="cpu",
                  include_symbolic_pianoroll=False):
    """Run the full DropItRight feature-extraction pipeline on one song.

    Returns a Music_info instance with both the baseline symbolic fields
    (vocal_info piano-roll, bpm, etc. -- populated by the existing
    segment_transcription path where available) and the new Indian-music /
    embedding fields populated.
    """
    title = title or os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Beat / downbeat tracking (TCN Carnatic, madmom fallback)
    beat_result = beat_tracking.track_beats(audio_path)
    beat_times, downbeat_start, rhythm, bpm = beat_result.as_tuple()

    # 2. Phrase-aligned segmentation
    try:
        segments_by_scale = segmentation.segment_phrases(
            beat_times, downbeat_start, rhythm
        )
    except Exception as exc:
        logger.warning(
            "Phrase segmentation failed (%s) -- falling back to fixed windows", exc
        )
        import soundfile as sfmod
        info = sfmod.info(audio_path)
        segments_by_scale = segmentation.segment_phrases_fixed_window(
            info.duration
        )

    # 3. Global (whole-song) features
    tonic_hz = None
    raga_info = {"raga": None, "confidence": None}
    try:
        tonic_hz = indian_features.extract_tonic(audio_path)
    except Exception as exc:
        logger.warning("Tonic extraction failed for %s: %s", audio_path, exc)

    try:
        raga_info = indian_features.extract_raga(audio_path=audio_path)
    except Exception as exc:
        logger.warning("Raga recognition failed for %s: %s", audio_path, exc)

    mert_embedding = None
    try:
        mert_embedding = global_embeddings.extract_mert_embedding(audio_path, device=device)
    except Exception as exc:
        logger.warning("MERT embedding failed for %s: %s", audio_path, exc)

    global_lyrics = None
    try:
        global_lyrics = global_embeddings.extract_lyrics(audio_path, device=device)
    except Exception as exc:
        logger.warning("Lyrics transcription failed for %s: %s", audio_path, exc)

    # 4. Per-segment features
    all_segments = []
    for duration_class, segments in segments_by_scale.items():
        for seg in segments:
            seg_wav = os.path.join(
                tmp_dir, f"{title}_{duration_class}_{seg.start:.2f}.wav"
            )
            seg_record = seg.as_dict()
            try:
                _render_segment_wav(audio_path, seg.start, seg.end, seg_wav)

                cae_embedding = None
                try:
                    cae_embedding = indian_features.extract_cae_embedding(
                        seg_wav, device=device
                    ).tolist()
                except Exception as exc:
                    logger.warning("CAE embedding failed for segment %s: %s", seg_wav, exc)

                melodysim_embedding = None
                try:
                    melodysim_embedding = melodysim_embeddings.extract_melodysim_embedding(
                        seg_wav, device=device
                    )
                except NotImplementedError:
                    pass  # melodysim model not wired up yet -- see melodysim_embeddings.py
                except Exception as exc:
                    logger.warning("melodysim embedding failed for %s: %s", seg_wav, exc)

                lyrics_slice = None
                if global_lyrics is not None:
                    lyrics_slice = global_embeddings.slice_lyrics(
                        global_lyrics, seg.start, seg.end
                    )

                seg_record.update({
                    "cae_embedding": cae_embedding,
                    "melodysim_embedding": melodysim_embedding,
                    "lyrics_slice": lyrics_slice,
                })
            finally:
                if os.path.exists(seg_wav):
                    os.remove(seg_wav)

            all_segments.append(seg_record)

    music_info = Music_info(
        title=title,
        bpm=int(bpm) if bpm else None,
        rhythm=int(rhythm) if rhythm else None,
        downbeat_start=float(downbeat_start) if downbeat_start is not None else None,
        beat_times=list(beat_times) if beat_times is not None else None,
        beat_track_source=beat_result.source,
        tonic_hz=tonic_hz,
        raga=raga_info.get("raga"),
        raga_confidence=raga_info.get("confidence"),
        global_mert_embedding=mert_embedding.tolist() if mert_embedding is not None else None,
        global_lyrics=global_lyrics,
        segments=all_segments,
    )

    # 5. Optional: baseline symbolic piano-roll signal (demucs + AST vocal
    # transcription + bar quantization), kept as an additional fused signal
    # per plan.md item 5 rather than replacing it. Heavy (demucs/AST/madmom),
    # so it's opt-in.
    if include_symbolic_pianoroll:
        try:
            from baseline_segment_transcription import segment_transcription
            import jsonpickle

            baseline_json_path = segment_transcription(audio_path)
            with open(baseline_json_path, "r", encoding="utf-8") as f:
                baseline_info = jsonpickle.decode(f.read())
            music_info.vocal_info = baseline_info.get("vocal_info")
        except Exception as exc:
            logger.warning(
                "Symbolic piano-roll extraction failed for %s: %s", audio_path, exc
            )

    return music_info
