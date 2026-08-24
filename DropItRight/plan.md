# Indian Music Segment-Level Similarity Pipeline — Plan

Extension of the MIPPIA `Music-Plagiarism-Detection` baseline pipeline for Indian/regional
music, using `compIAM` tools for Indian-music-specific features. Goal: a "Turnitin for music" —
whole-song similarity + lyrics matching to find top-K candidate songs, then fine-grained
segment-level report showing which parts of a song are similar to which parts of matched songs.

## High-level flow

```
A) INDEXING (batch, run once per reference song)
   raw audio -> process_song() -> SongProfile JSON -> saved to reference DB

B) QUERY (per user upload)
   raw audio -> process_song() -> SongProfile JSON
       -> Stage 1: global match against all reference SongProfiles -> top-K songs
       -> Stage 2: segment-level match against only those K songs -> per-segment report
```

`process_song()` is the single function both indexing and query paths call. It replaces/extends
`segment_transcription.py` from the baseline.

## Segmentation: phrase-aligned instead of fixed windows

**Tracker choice: TCN Carnatic beat tracker (`compiam.rhythm.meter.tcn_carnatic.TCNTracker`) as
primary, madmom (existing baseline tracker) as fallback.**

- `AksharaPulseTracker` gives tala-pulse-level resolution but is classical DSP tuned to strict
  tala structure — can drift on regional/film songs that don't keep tala discipline.
- `TCNTracker` is more robust across the wider tempo/genre range seen in regional film music,
  and already outputs beats + downbeat structure compatible with what the baseline's
  `madmom` trackers currently feed into `wav_quantizing()`.

**Segmentation rule:**
1. Run `TCNTracker` -> beat times + downbeat positions (bars).
2. Group consecutive bars into phrase segments, snapping segment boundaries to the nearest
   downbeat rather than a raw time cut (same idea as baseline's `infos_to_startpoint` /
   4-bar grouping, just with a Carnatic-tuned tracker instead of Western-tuned madmom).
3. Instead of fixed 3s/5s/7s windows, define segments as **N bars**, where N is chosen per
   song from its detected tempo so resulting duration lands near the 3/5/7s targets. Multi-scale
   segments are preserved, but every boundary falls on a real phrase edge instead of mid-note.
4. Fallback: if TCN's beat confidence is low or it errors, fall back to fixed-window
   segmentation so the pipeline never blocks on a bad beat estimate.

This replaces "segment audio into 3s/5s/7s" with "segment audio into phrase-aligned chunks that
approximate those durations" — same downstream shape (list of segments per song), so it drops
into the existing 2-level match architecture without changing the matcher's interface.

## Three Indian features for v1

**1. Tonic Identification — `compiam.melody.tonic_identification.tonic_multipitch.TonicIndianMultiPitch`**
Role: normalization, not a matching feature by itself. Indian melody comparison must be
tonic-relative (same tune in a different key/tonic should still match). Run once per song
(global stage), use it to normalize pitch curves before segment-level matching. Cheap
(essentia, DSP-only, no GPU) — belongs in the global-embedding stage.

**2. Raga Recognition — `compiam.melody.raga_recognition.deepsrgm.DEEPSRGM`**
Role: coarse filter tag at the global level, feeding the Stage-1 top-K step. Two songs (or
segments) in unrelated ragas are unlikely to be melodically plagiarized in the classical sense —
use the predicted raga as cheap metadata to prune/re-rank candidates before the expensive
segment-level match runs (same spirit as the BPM-ratio gate already in baseline
`compare.py`'s `calculate_metric_optimized`).
Caveat: DEEPSRGM's raga set is a fixed ~40-raga mapping trained on Carnatic art music. On
filmy/regional songs, treat its output as a **soft signal** (down-weight in scoring) rather than
a hard filter — don't drop candidates solely on raga mismatch, since film music often doesn't
map cleanly onto any raga.

**3. Melodic Pattern Embeddings — `compiam.melody.pattern.sancara_search.CAEWrapper` (CAE-Carnatic)**
Role: the core Indian melodic embedding for segment-level matching, alongside MERT and the
existing melodysim embeddings. Built specifically for "find similar melodic phrases" — the most
direct fit for the segment-matching task. Extract CAE features per segment (from CQT, tonic-
normalized), use as one of the embedding channels compared at segment level (cosine/DTW
distance), combined with MERT + melodysim scores in the final per-segment fused score.

## Pipeline stages, mapped onto the baseline codebase

**1. Source separation** — keep `segment_transcription.py`'s demucs step as-is
(vocals/other/drums/bass). No change needed.

**2. Beat/downbeat tracking — replacement point**
- Baseline: `madmom` `DBNBeatTrackingProcessor` / `DBNDownBeatTrackingProcessor`
  (`segment_transcription.py` lines ~20-26), feeding `wav_quantizing()`.
- New: swap in `TCNTracker` here, madmom kept as fallback (try TCN first, catch low-confidence/
  failure -> fall back to madmom). Same slot in the code; output shape stays compatible with
  what `wav_quantizing` / `infos_to_startpoint` expect downstream.

**3. Segmentation — new logic, new function**
- Baseline builds bar-groups via `infos_to_startpoint()` / `refine_breakpoints_custom()` in
  `compare_utils.py` (fixed 4-bar windows).
- New: `segment_phrases(beat_times, downbeats, target_durations=[3,5,7])` groups bars into
  phrase-aligned chunks near each target duration. Produces the segment boundary list that
  embeddings, lyrics, and Indian features all slice audio against. Called right after beat
  tracking, before the rest of `segment_transcription.py` runs.

**4. Global (whole-song) feature extraction — new module, runs once per song**
- MERT embedding: whole vocal+mix audio -> embedding vector.
- Indic ASR lyrics: whole vocal stem -> transcript.
- Tonic (`TonicIndianMultiPitch`): whole audio -> single tonic Hz value, stored and used to
  normalize all segment-level pitch features for this song.
- Raga tag (`DEEPSRGM`): whole audio -> raga label + confidence, stored as metadata for
  Stage-1 filtering.

**5. Segment-level feature extraction — extends existing per-bar processing**
- Existing: vocal MIDI transcription (AST) -> quantized piano-roll per segment
  (`vocal_midi2note`, `quantize` in `utils.py`), consumed by `compare.py`'s
  `TestDataset` / `infos_to_pianorolls`.
- New, added per segment: CAE-Carnatic embedding (tonic-normalized pitch curve via
  `CAEWrapper.extract_features`), melodysim embedding, per-segment lyric slice (cut from the
  global ASR transcript by segment timestamps rather than re-transcribing).
- Keep the symbolic piano-roll path as one more fused signal rather than replacing it — still a
  valid (if lossy) melodic contour cue.

**6. Data model — extend `Music_info`**
Add fields to `music_info.py`'s `Music_info`: `tonic`, `raga`, `raga_confidence`,
`global_mert_embedding`, `global_lyrics`. Change segment representation from the current flat
`vocal_info` dict-by-bar-index to `segments: List[Segment]`, each carrying:
`{start, end, duration_class(3/5/7), cae_embedding, melodysim_embedding, piano_roll(existing),
lyrics_slice}`. This is the JSON schema written by `process_song()` and read back by both the
indexer and the query-time matcher (same role `sav_path` / `jsonpickle` plays today).

**7. Reference DB**
Replaces the flat `covers80/*.json` glob (`compare.py` lines 14-15) with an actual index: a
vector index (FAISS or similar) over `global_mert_embedding` (+ raga/tonic as filterable
metadata) for Stage 1, and per-song segment JSONs on disk (or in the same store) fetched lazily
only for the Stage-1 top-K songs in Stage 2. `TestDataset2` in `compare.py` becomes "load
segment JSONs for these K song-ids" instead of "glob every JSON in the folder."

**8. Stage 1: global match — new, sits before existing `compare.py` logic**
Query song's `global_mert_embedding` (+ lyrics similarity) -> ANN search over reference DB ->
top-K candidate songs, re-ranked/down-weighted by raga match (soft signal, not hard filter).
Output: list of K song-ids, replacing "compare against everything in `covers80/`" in current
`get_one_result`.

**9. Stage 2: segment match — extends `compare.py`'s existing matcher**
For only the K candidate songs, run segment-vs-segment matching. Existing
`calculate_metric_optimized` (piano-roll DTW/correlation) stays as one term; add cosine/DTW
distance on `cae_embedding` and `melodysim_embedding` as additional terms, fused into one
`final_metric` per segment pair the same way `compare.py` already fuses
pitch_score/correlation/bpm_ratio (~line 417). `CompareHelper` heap logic (top-N segment
matches) stays unchanged — genre-agnostic bookkeeping.

**10. Reporting — extends `inference.py`'s `result_formatting`**
Same shape as today (rank, score, song_title, time ranges), but grouped as: overall song-level
similarity (from Stage 1) + list of matched segments with per-segment score breakdown
(melody/lyrics/rhythm sub-scores) so the report can show *why* a segment matched, not just that
it did.

## Proposed new module layout

```
Music-Plagiarism-Detection/
  indian_features.py       # tonic, raga, CAE-Carnatic wrappers (calls into compIAM)
  segmentation.py          # segment_phrases(), replaces fixed-window logic
  beat_tracking.py         # TCNTracker + madmom fallback wrapper
  global_embeddings.py     # MERT + indic ASR calls
  process_song.py          # orchestrates: separation -> beat track -> segment -> global feats
                            #   -> segment feats -> Music_info JSON
  reference_db.py          # ANN index build/query (Stage 1), segment JSON store (Stage 2 fetch)
  compare.py               # extended: add CAE/melodysim terms to calculate_metric_optimized
  music_info.py            # extended schema
```

Every baseline file's existing job stays intact and additive — the DTW/correlation math in
`compare_utils.py` doesn't need to change, it just gets more input channels.

## Open items / next steps
- Scaffold `process_song.py` and `segmentation.py` (the two genuinely new orchestration pieces).
- Work out fusion weights for Stage 2 scoring (how CAE/melodysim/piano-roll/lyrics combine into
  `final_metric`).
- Decide ANN index tech for the reference DB (FAISS vs alternatives) and index update strategy
  as new reference songs are added.
