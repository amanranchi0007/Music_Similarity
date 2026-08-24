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


def _get_raga_model():
    global _raga_model
    if _raga_model is None:
        import compiam
        _raga_model = compiam.load_model("melody:deepsrgm")
    return _raga_model


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


def extract_raga(pitch_path=None, tonic_hz=None, audio_path=None, input_sr=44100, k=5):
    """Whole-song raga prediction.

    DEEPSRGM's own feature extractor (get_features) expects either raw audio or
    precomputed pitch/tonic paths -- wire whichever is available. Returned as a
    soft signal: (raga_label, confidence), not a hard filter (see plan.md caveat
    about DEEPSRGM's ~40-raga mapping not covering regional/film music cleanly).
    """
    model = _get_raga_model()
    try:
        features = model.get_features(
            input_data=audio_path,
            input_sr=input_sr,
            pitch_path=pitch_path,
            tonic_path=None,
            k=k,
        )
        prediction = model.predict(features)
        # DEEPSRGM.predict returns majority-voted raga id(s); shape/format
        # depends on compiam version -- normalize defensively.
        if isinstance(prediction, (list, tuple, np.ndarray)) and len(prediction) > 0:
            raga_label = prediction[0]
            confidence = float(prediction[1]) if len(prediction) > 1 else None
        else:
            raga_label, confidence = prediction, None
        return {"raga": raga_label, "confidence": confidence}
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
