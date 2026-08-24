"""
Adapter around AMAAI-Lab's MelodySim (https://github.com/AMAAI-Lab/MelodySim,
https://huggingface.co/amaai-lab/MelodySim), a triplet-trained Siamese network
for melody-aware plagiarism similarity.

The `SiameseNet` architecture below is vendored verbatim from that repo's
`model/siamese_net.py` (not pip-installable, and the repo's own loading path
via `LightningSiameseNet` pulls in `lightning` + a training dataloader we
don't need for inference) so this module has no dependency beyond torch/
transformers/librosa, matching the rest of DropItRight.

IMPORTANT semantics: unlike the CAE-Carnatic embedding (compared by cosine
similarity), MelodySim's embedding space is triplet-margin trained with
*Euclidean distance* -- "same songs are labelled with 0 in training (means
low distance)" per the original repo. fusion.py's melodysim scoring must use
a distance-based similarity, not cosine (see fusion._melodysim_similarity).

Setup:
    1. Download the checkpoint:
       https://huggingface.co/amaai-lab/MelodySim/resolve/main/siamese_net_20250328.ckpt
    2. export DROPITRIGHT_MELODYSIM_CKPT=/path/to/siamese_net_20250328.ckpt
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import librosa

logger = logging.getLogger(__name__)

MELODYSIM_CKPT = os.environ.get("DROPITRIGHT_MELODYSIM_CKPT")
# MelodySim was trained against MERT-v1-95M specifically (not the v1-330M
# DropItRight uses for the whole-song global embedding in global_embeddings.py)
# -- the checkpoint's weights expect that exact feature distribution.
MELODYSIM_MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MELODYSIM_EMB_DIM = 128
MELODYSIM_SAMPLE_RATE = 44100

_melodysim_model = None
_mert_model = None
_mert_processor = None


# --- vendored from AMAAI-Lab/MelodySim model/siamese_net.py -----------------

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return self.relu(out)


class _SiameseNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layer1 = _ResidualBlock(3072, 512)
        self.layer2 = _ResidualBlock(512, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # length-invariant: any segment duration works
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------------------------------------------------------


def _get_mert(device="cpu"):
    global _mert_model, _mert_processor
    if _mert_model is None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        logger.info("Loading MelodySim's MERT model %s", MELODYSIM_MERT_MODEL_ID)
        _mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(MELODYSIM_MERT_MODEL_ID)
        _mert_model = AutoModel.from_pretrained(
            MELODYSIM_MERT_MODEL_ID, trust_remote_code=True
        ).to(device)
        _mert_model.eval()
    return _mert_model, _mert_processor


def _load_model(device="cpu"):
    global _melodysim_model
    if _melodysim_model is None:
        if not MELODYSIM_CKPT:
            raise RuntimeError(
                "DROPITRIGHT_MELODYSIM_CKPT is not set -- point it at "
                "siamese_net_20250328.ckpt from "
                "https://huggingface.co/amaai-lab/MelodySim before running "
                "the real pipeline. See melodysim_embeddings.py."
            )
        logger.info("Loading MelodySim checkpoint %s", MELODYSIM_CKPT)
        checkpoint = torch.load(MELODYSIM_CKPT, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Checkpoint was saved from LightningSiameseNet, whose only relevant
        # submodule for inference is `siamese_net.*` (the classifier head and
        # optimizer state aren't needed for embedding extraction).
        siamese_state = {
            k[len("siamese_net."):]: v
            for k, v in state_dict.items()
            if k.startswith("siamese_net.")
        }
        if not siamese_state:
            raise RuntimeError(
                f"No 'siamese_net.*' keys found in {MELODYSIM_CKPT} -- "
                "checkpoint format doesn't match what was expected "
                "(LightningSiameseNet state_dict)."
            )
        model = _SiameseNet(embedding_dim=MELODYSIM_EMB_DIM)
        model.load_state_dict(siamese_state)
        model.to(device)
        model.eval()
        _melodysim_model = model
    return _melodysim_model


def extract_melodysim_embedding(segment_audio_path, device="cpu"):
    """Per-segment MelodySim embedding (128-dim). Compare with
    fusion._melodysim_similarity (Euclidean-distance-based), NOT cosine --
    see module docstring."""
    model = _load_model(device=device)
    mert_model, mert_processor = _get_mert(device=device)

    waveform, _ = librosa.load(segment_audio_path, sr=MELODYSIM_SAMPLE_RATE, mono=True)
    inputs = mert_processor(waveform, sampling_rate=MELODYSIM_SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        hidden_states = mert_model(**inputs, output_hidden_states=True).hidden_states

        # Exact feature prep the checkpoint was trained on (module.py's
        # inference_pairs): take every 3rd hidden layer starting at index 2,
        # average-pool 10 frames -> 1 to downsample time, concat the 4
        # selected layers along the channel dim -> [1, 3072, frames].
        time_reduce = nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False)
        selected = hidden_states[2::3]
        pooled = [time_reduce(h.permute(0, 2, 1)).permute(0, 2, 1) for h in selected]
        stacked = torch.stack(pooled, dim=1)  # [1, num_layers, frames, layer_dim]
        batch, num_layers, num_frames, layer_dim = stacked.shape
        if num_layers != 4 or layer_dim != 768:
            raise RuntimeError(
                f"Unexpected MERT hidden_states shape for MelodySim "
                f"(num_layers={num_layers}, layer_dim={layer_dim}); expected "
                f"4 layers of dim 768 -- check MELODYSIM_MERT_MODEL_ID matches "
                f"the checkpoint's training config."
            )
        stacked = stacked.permute(0, 1, 3, 2)  # [1, num_layers=4, layer_dim=768, frames]
        features = torch.cat([stacked[:, i] for i in range(num_layers)], dim=1)  # [1, 3072, frames]

        embedding = model(features)  # [1, 128]

    return embedding.squeeze(0).cpu().numpy()
