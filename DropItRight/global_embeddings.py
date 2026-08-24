"""
Whole-song global features: MERT audio embedding + Indic ASR lyrics transcript.

These feed Stage-1 (global) similarity search over the reference DB. Both are
HuggingFace-hosted models; weights download on first use and cache under HF's
default cache dir (~/.cache/huggingface). Swap MODEL ids via env vars if you
want a different checkpoint on the GPU server without touching code.
"""

import os
import logging
import numpy as np
import librosa
import torch

logger = logging.getLogger(__name__)

MERT_MODEL_ID = os.environ.get("DROPITRIGHT_MERT_MODEL", "m-a-p/MERT-v1-330M")
# NOTE: pick and verify the exact ASR model id for your target languages before
# relying on this for real matching -- "ai4bharat/indicconformer_stt_multilingual"
# does not exist on the HF Hub; AI4Bharat ships per-language IndicConformer
# checkpoints (e.g. ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large) rather
# than one multilingual id. whisper-large-v3 is used here as a working
# multilingual default that covers major Indian languages reasonably.
INDIC_ASR_MODEL_ID = os.environ.get(
    "DROPITRIGHT_INDIC_ASR_MODEL", "openai/whisper-large-v3"
)

_mert_model = None
_mert_processor = None
_asr_pipeline = None


def _get_mert(device="cpu"):
    global _mert_model, _mert_processor
    if _mert_model is None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        logger.info("Loading MERT model %s", MERT_MODEL_ID)
        _mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_MODEL_ID, trust_remote_code=True
        )
        _mert_model = AutoModel.from_pretrained(
            MERT_MODEL_ID, trust_remote_code=True
        ).to(device)
        _mert_model.eval()
    return _mert_model, _mert_processor


def extract_mert_embedding(audio_path, device="cpu", pooling="mean"):
    """Whole-song MERT embedding, mean-pooled across all transformer layers
    and time. Returns a single fixed-size vector for ANN indexing."""
    model, processor = _get_mert(device=device)
    target_sr = processor.sampling_rate

    # librosa/soundfile-based load -- avoids torchaudio.load's dependency on
    # torchcodec, which isn't installed by default in newer torchaudio releases.
    waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)

    inputs = processor(
        waveform, sampling_rate=target_sr, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple(num_layers+1) of [1, T, C] -- stack, then mean over
    # layers and time to get one embedding per song (MERT layer-aggregation
    # is task-dependent; mean-pooling is a solid default for retrieval).
    all_layers = torch.stack(outputs.hidden_states, dim=0).squeeze(1)  # [L, T, C]
    embedding = all_layers.mean(dim=(0, 1)).cpu().numpy()
    return embedding


def _get_asr_pipeline(device="cpu"):
    global _asr_pipeline
    if _asr_pipeline is None:
        from transformers import pipeline

        logger.info("Loading Indic ASR model %s", INDIC_ASR_MODEL_ID)
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=INDIC_ASR_MODEL_ID,
            device=0 if device == "cuda" else -1,
        )
    return _asr_pipeline


def extract_lyrics(vocal_audio_path, device="cpu"):
    """Whole-song lyrics transcript from the isolated vocal stem, with
    word/segment-level timestamps so segment-level lyric slices (see
    process_song.py) can be cut from this single transcription instead of
    re-running ASR per segment.

    Loads audio via librosa (libsndfile, no external binary) and passes a raw
    array to the pipeline instead of a file path -- HF's ASR pipeline shells
    out to ffmpeg to decode paths, which we can't rely on being installed on
    every target machine.
    """
    asr = _get_asr_pipeline(device=device)
    target_sr = asr.feature_extractor.sampling_rate
    waveform, _ = librosa.load(vocal_audio_path, sr=target_sr, mono=True)
    result = asr({"array": waveform, "sampling_rate": target_sr}, return_timestamps=True)
    return {
        "text": result.get("text", ""),
        "chunks": result.get("chunks", []),  # [{"text":..., "timestamp": (start,end)}]
    }


def slice_lyrics(global_lyrics, seg_start, seg_end):
    """Cut the per-segment lyric slice out of the whole-song transcript's
    timestamped chunks, avoiding a second ASR pass per segment."""
    chunks = global_lyrics.get("chunks", [])
    words = []
    for chunk in chunks:
        start, end = chunk.get("timestamp", (None, None))
        if start is None:
            continue
        if start < seg_end and end > seg_start:
            words.append(chunk["text"])
    return " ".join(words).strip()
