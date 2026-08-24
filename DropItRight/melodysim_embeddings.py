"""
Thin adapter around your existing melody-similarity embedding model.

This repo doesn't vendor that model -- it's the one from your original melody-sim
pipeline. Point DROPITRIGHT_MELODYSIM_CKPT at its checkpoint and fill in
`_load_model` / `extract_melodysim_embedding` with the real inference call.
Kept as a separate module (rather than inlined in process_song.py) so swapping
in the real model later is a one-file change.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

MELODYSIM_CKPT = os.environ.get("DROPITRIGHT_MELODYSIM_CKPT")

_melodysim_model = None


def _load_model(device="cpu"):
    global _melodysim_model
    if _melodysim_model is None:
        if not MELODYSIM_CKPT:
            raise RuntimeError(
                "DROPITRIGHT_MELODYSIM_CKPT is not set -- point it at your "
                "existing melody-sim model checkpoint before running the "
                "real pipeline. See melodysim_embeddings.py."
            )
        # TODO: replace with the actual loading call for your melodysim model,
        # e.g.:
        #   _melodysim_model = torch.load(MELODYSIM_CKPT, map_location=device)
        #   _melodysim_model.eval()
        raise NotImplementedError(
            "Wire up your melodysim model's load routine here."
        )
    return _melodysim_model


def extract_melodysim_embedding(segment_audio_path, device="cpu"):
    """Per-segment melody-similarity embedding, one of the fused signals in
    Stage-2 segment matching alongside the CAE-Carnatic and piano-roll terms.
    """
    model = _load_model(device=device)
    # TODO: replace with the real forward pass for your melodysim model.
    raise NotImplementedError(
        "Wire up your melodysim model's inference call here."
    )
