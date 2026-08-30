"""
Indian-music-specific features, all sourced from compIAM (https://github.com/MTG/compIAM).

Three features for v1, per plan.md:

1. Tonic Identification (TonicIndianMultiPitch) -- global, DSP-only (essentia), used to
   normalize pitch curves before segment-level matching. No pretrained weights.
2. Raga Recognition (DEEPSRGM) -- global, soft coarse-filter tag for Stage-1 candidate
   pruning/re-ranking. Pretrained weights auto-downloaded by compiam.load_model.
3. Melodic pattern embeddings (CAE-Carnatic / sancara_search) -- per-segment embedding,
   the core Indian melodic-similarity signal used in Stage-2 segment matching.
   Pretrained weights auto-downloaded by compiam.load_model.

compIAM handles model weight downloading/caching itself via compiam.load_model(), so no
model files need to be bundled in this repo -- they land in compIAM's WORKDIR on first use.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_tonic_model = None
_raga_model = None
_cae_model = None


def _get_tonic_model():
    global _tonic_model
    if _tonic_model is None:
        from compiam.melody.tonic_identification.tonic_multipitch import (
            TonicIndianMultiPitch,
        )
        _tonic_model = TonicIndianMultiPitch()
    return _tonic_model


def _get_raga_model(device="cpu"):
    global _raga_model
    if _raga_model is None:
        import compiam
        # Without an explicit device, DEEPSRGM's own __init__ defaults to
        # "cuda" (i.e. cuda:0) whenever *any* GPU is visible, ignoring
        # whichever --device this process was actually told to use -- on a
        # shared GPU box that silently piles every process's raga model onto
        # device 0 and can OOM it even when the rest of the pipeline is on
        # cuda:1/cuda:2. Route it explicitly instead.
        _raga_model = compiam.load_model("melody:deepsrgm", device=device)
    return _raga_model


def unload_raga_model():
    """Free DEEPSRGM's GPU memory as soon as raga extraction is done for a
    song. Its forward pass always runs a fixed batch of 200 random 5000-
    sample subsequences through an LSTM (hidden_size=768) -- a constant
    ~15GB single allocation regardless of song length, unrelated to whatever
    else (MERT/CAE/MelodySim/Whisper) is about to load onto the same device.
    Called right after extract_raga() in process_song.py so that allocation
    is released before the rest of the pipeline claims GPU memory, instead
    of everything sitting in VRAM at once."""
    global _raga_model
    if _raga_model is not None:
        del _raga_model
        _raga_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _get_cae_model(device="cpu"):
    global _cae_model
    if _cae_model is None:
        import compiam
        _cae_model = compiam.load_model("melody:cae-carnatic", device=device)
    return _cae_model


def extract_tonic(audio_path, input_sr=44100):
    """Whole-song tonic frequency (Hz). Used to normalize segment pitch curves."""
    model = _get_tonic_model()
    tonic_hz = model.extract(audio_path, input_sr=input_sr)
    return float(tonic_hz)


def extract_raga(pitch_path=None, tonic_hz=None, audio_path=None, input_sr=44100, k=5,
                  device="cpu", confidence_threshold=0.7):
    """Whole-song raga prediction.

    DEEPSRGM's own feature extractor (get_features) expects either raw audio or
    precomputed pitch/tonic paths -- wire whichever is available. Returned as a
    soft signal: (raga_label, confidence), not a hard filter (see plan.md caveat
    about DEEPSRGM's ~40-raga mapping not covering regional/film music cleanly).

    IMPORTANT: DEEPSRGM has no "not Carnatic" class -- model.predict() always
    forces the input into one of its ~10-40 trained ragas, no matter how little
    the audio actually resembles any of them (it *does* compute a majority-vote
    "confidence" internally and even logs "CONFUSED" below its own default
    threshold=0.6, but still returns the majority label regardless -- the
    threshold only gates a log message, not the return value). That's why raga
    labels were showing up on clearly non-Carnatic tracks. We replicate
    predict()'s own vote computation here (same model, same features) so we can
    actually withhold the label -- reporting raga=None -- when confidence is
    below confidence_threshold, instead of always forcing a guess.
    """
    import torch

    model = _get_raga_model(device=device)
    try:
        features = model.get_features(
            input_data=audio_path,
            input_sr=input_sr,
            pitch_path=pitch_path,
            tonic_path=None,
            k=k,
        )
        # Mirrors DEEPSRGM.predict()'s internals (majority vote across
        # subsequences) instead of calling predict() itself, for two reasons:
        # 1. predict() unconditionally sets CUDA_VISIBLE_DEVICES via its own
        #    `gpu` kwarg (default "-1"), which fights with the device this
        #    model/process was actually placed on.
        # 2. predict() never returns its vote fraction, only the forced label.
        with torch.no_grad():
            out = model.model.forward(torch.from_numpy(features).to(model.device).long())
        preds = torch.argmax(out, axis=-1)
        majority, _ = torch.mode(preds)
        majority = int(majority)
        votes = float(torch.sum(preds == majority)) / features.shape[0]

        if model.mapping is None:
            model.load_mapping(model.selected_ragas)
        raga_label = model.mapping[majority] if votes >= confidence_threshold else None
        return {"raga": raga_label, "confidence": votes}
    except Exception as exc:
        logger.warning("Raga recognition failed for %s: %s", audio_path, exc)
        return {"raga": None, "confidence": None}


def _to_numpy(x):
    """CAEWrapper.extract_features returns tensors that still require grad
    (the wrapper doesn't wrap its forward pass in torch.no_grad()), so a bare
    .numpy() raises. Detach defensively regardless of tensor/array input."""
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_cae_embedding(segment_audio_path, device="cpu"):
    """Per-segment melodic-pattern embedding (amplitude + phase vectors from the
    CAE-Carnatic complex autoencoder), the core signal for Stage-2 segment matching.

    NOTE: CAEWrapper.extract_features operates on a CQT computed from the segment's
    audio -- pass a path to a rendered single-segment wav (see process_song.py),
    not a full-song path with offsets, since the wrapper takes a file path.
    """
    model = _get_cae_model(device=device)
    amplitude, phase = model.extract_features(segment_audio_path)
    amplitude = _to_numpy(amplitude)
    phase = _to_numpy(phase)
    # Collapse to a fixed-size vector for nearest-neighbour / cosine comparison
    # downstream; keep both amplitude and phase since CAE similarity in the
    # original sancara_search work uses both channels.
    embedding = np.concatenate(
        [amplitude.mean(axis=0).ravel(), phase.mean(axis=0).ravel()]
    )
    return embedding


def normalize_pitch_to_tonic(pitch_hz, tonic_hz):
    """Convert absolute pitch (Hz) to tonic-relative cents, so melodies in
    different keys/tonics compare correctly. Standard IAM convention: 1200
    cents per octave relative to the tonic."""
    pitch_hz = np.asarray(pitch_hz, dtype=float)
    valid = pitch_hz > 0
    cents = np.zeros_like(pitch_hz)
    cents[valid] = 1200.0 * np.log2(pitch_hz[valid] / tonic_hz)
    return cents
