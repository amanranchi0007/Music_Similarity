# DropItRight

Segment-level music similarity checker & reporter for Indian/regional music —
"Turnitin for music." Extends the MIPPIA `Music-Plagiarism-Detection` baseline
pipeline with Indian-music-specific features from `compIAM`, phrase-aligned
segmentation, and a two-level (global → segment) retrieval flow.

Full design rationale: see `plan.md` in the parent `TCS_ILP` folder (copy it into
this folder too if you want it on the GPU server).

## What's in here

```
Baseline (copied from Music-Plagiarism-Detection, mostly untouched):
  music_info.py                  Music_info schema — extended with new fields (see below)
  compare_utils.py               piano-roll DTW/correlation math — untouched
  compare.py                     baseline segment matcher (symbolic piano-roll path)
  utils.py                       MIDI/quantization helpers
  wav_quantizer.py               Beat-Transformer-based quantizer (only used if you
                                  opt into include_symbolic_pianoroll=True)
  baseline_segment_transcription.py   original segment_transcription(), renamed
  inference.py                   original top-level inference() entrypoint
  ml_models/AST/                 vocal transcription model (weights included, ~17MB)
  ml_models/DilatedTransformer*.py    Beat-Transformer model code (checkpoint NOT
                                  included — see ml_models/Beat-Transformer/checkpoint/README.txt)

New (DropItRight extension):
  beat_tracking.py               TCN Carnatic beat/downbeat tracker (compIAM), madmom fallback
  segmentation.py                phrase-aligned segmentation (replaces fixed 3s/5s/7s windows)
  indian_features.py             Tonic, Raga (DEEPSRGM), CAE-Carnatic melodic embeddings (compIAM)
  global_embeddings.py           MERT whole-song embedding + Indic ASR lyrics transcript
  melodysim_embeddings.py        AMAAI-Lab MelodySim (Siamese/triplet MERT-95M model) — wired,
                                  needs DROPITRIGHT_MELODYSIM_CKPT set (see below)
  process_song.py                orchestrates all of the above into one Music_info profile
  reference_db.py                FAISS-based Stage-1 index + Stage-2 lazy segment-profile store
  fusion.py                      Stage-2 per-segment score fusion (cae + melodysim + lyrics [+ piano_roll])
  build_index.py                 CLI: batch-index a folder of reference songs
  query.py                       CLI: run a query song through Stage 1 + Stage 2, print report
  visualize.py                   CLI: render HTML reports (DB overview, or one query's
                                  per-feature segment heatmaps + match timeline)
```

## Setup on the GPU server

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `compiam` and the HuggingFace models (`m-a-p/MERT-v1-330M`, the Indic ASR model)
  download their weights on first use — make sure the server has internet access
  the first time you run, or pre-warm the caches and copy `~/.cache/huggingface`
  and compIAM's model cache (`compiam.data.WORKDIR`) over separately.
- `madmom` on PyPI can lag your Python version; if `pip install madmom` fails,
  install from source: `pip install git+https://github.com/CPJKU/madmom`
- `essentia` (required by tonic extraction) is easiest via conda on some
  platforms if the pip wheel isn't available for your Python/OS combo.

## Before running for real

**`melodysim_embeddings.py` needs a checkpoint set.** It's wired to
[AMAAI-Lab/MelodySim](https://huggingface.co/amaai-lab/MelodySim), a
triplet-trained Siamese network over MERT-v1-95M features. Download the
checkpoint and point the env var at it:

```bash
wget https://huggingface.co/amaai-lab/MelodySim/resolve/main/siamese_net_20250328.ckpt
export DROPITRIGHT_MELODYSIM_CKPT=/path/to/siamese_net_20250328.ckpt
```

Without it set, `process_song.py` catches the resulting `RuntimeError` and
continues — CAE + lyrics alone will produce a (partial) similarity score —
but set it before trusting the fused numbers, since melodysim carries the
single largest fusion weight (0.35, tied with CAE).

Note: MelodySim's embedding space is trained on **Euclidean distance**, not
cosine (same-song pairs are trained toward distance ≈ 0) — `fusion.py`
accounts for this with a dedicated `_melodysim_similarity` (distance → bounded
similarity via `1/(1+distance)`), separate from CAE's cosine scoring. It also
loads its own MERT-v1-95M instance (the checkpoint was trained against that
exact model), independent of the MERT-v1-330M used for the whole-song global
embedding elsewhere in this pipeline — expect two MERT models resident in
memory/VRAM when both run.

**`--symbolic-pianoroll` / `include_symbolic_pianoroll=True`** re-enables the
baseline's demucs+AST+Beat-Transformer piano-roll path as an extra fused signal.
It needs `ml_models/Beat-Transformer/checkpoint/fold_4_trf_param.pt`, which is
NOT bundled here (it's a 200MB+ file from the original MIPPIA repo and is
redundant with the new TCN-Carnatic-based beat tracking for the primary path).
Fetch it separately if you want that signal too; otherwise leave the flag off.

## Usage

```bash
# 1. Index your reference songs (run once, or whenever you add new songs)
python build_index.py --audio-dir /path/to/reference_songs --db-dir ./reference_db --device cuda

# 2. Query a user-uploaded song
python query.py --audio /path/to/uploaded_song.wav --db-dir ./reference_db --top-k 5 --device cuda
```

`query.py` prints a JSON report: overall global similarity score per matched
reference song, plus the top segment-pair matches (query time range ↔
reference time range ↔ per-segment fused score) — the data a report UI would
render as "these N seconds of your song look like these N seconds of song X."

### Visualizing

`visualize.py` is a separate CLI (doesn't touch the indexing/query path) that
renders self-contained HTML reports — open the file in a browser, or scp it
off the GPU server:

```bash
# DB overview: song-to-song global similarity matrix + raga/tonic table
python visualize.py db --db-dir ./reference_db --out report_db.html

# Query report: Stage-1 candidate bar chart, per-feature segment similarity
# heatmaps (cae / melodysim / lyrics / fused) for the top-matched reference
# song, and a Turnitin-style match timeline across all segment scales
python visualize.py query --audio tests/audio/query_song.wav \
  --db-dir ./reference_db --top-k 5 --duration-class 5 --device cuda \
  --out report_query.html

# reuse an already-computed profile instead of recomputing MERT/CAE/ASR:
python query.py --audio song.wav --db-dir ./reference_db --save-profile query_profile.json
python visualize.py query --profile query_profile.json --db-dir ./reference_db \
  --duration-class 5 --out report_query.html
```

`--duration-class` picks which segment scale (`3`, `5`, `7`, or `whole`) the
heatmaps are drawn at — mixing scales in one matrix doesn't make sense since
segment counts differ; the match timeline plot, unlike the heatmaps, does
pool all scales together since it's just drawing time ranges.

## Known gaps / next steps (tracked from plan.md)

- Wire up `melodysim_embeddings.py` with your actual model.
- `fusion.py`'s lyrics similarity is an n-gram (BLEU-style) metric — good at
  catching near-verbatim/copied lines, weak on paraphrased or transliterated
  lyrics with the same meaning but different words. Swap for an
  embedding-based multilingual similarity if that recall gap matters for
  your use case.
- `DEEPSRGM`'s raga vocabulary (~40 Carnatic ragas) doesn't cover most
  regional/film music cleanly. `reference_db.search_global` now applies a
  soft re-rank boost (`raga_boost`, default 0.05) to same-raga candidates —
  pass `query_raga=None` (the default) to disable it entirely until you've
  validated raga predictions are reliable enough on your actual regional-song
  corpus to be worth the nudge.
- No auth/API layer yet — `build_index.py`/`query.py` are CLI-only; wrap
  `process_song`/`ReferenceDB`/`fusion` in a service (FastAPI, etc.) for the
  actual upload-and-report product surface.
