"""
Reference database for the two-level retrieval flow (plan.md, item 7).

Replaces the baseline's flat `glob(covers80/*.json)` (compare.py) with:
  - a FAISS index over each song's global_mert_embedding, for Stage-1 ANN search,
    with raga/tonic kept as filterable metadata,
  - per-song segment-profile JSON files on disk, fetched lazily and only for the
    Stage-1 top-K songs during Stage 2 (TestDataset2 in compare.py should read
    from here instead of globbing every file up front).

Kept dependency-light (faiss-cpu + jsonpickle) so it runs the same on the GPU
server as anywhere else; swap to faiss-gpu / a real vector DB (e.g. Milvus,
Qdrant) later without changing the call sites below.
"""

import os
import json
import hashlib
import logging
import numpy as np
import jsonpickle

logger = logging.getLogger(__name__)


def _stable_id(song_id):
    """Deterministic int64 id for a song_id string, stable across processes
    (unlike Python's hash()) so FAISS add_with_ids/remove_ids can target the
    same row for a song on every run, letting re-indexing replace rather than
    duplicate an entry."""
    digest = hashlib.sha1(song_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)  # 60 bits, safely within int64 range


class ReferenceDB:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.profiles_dir = os.path.join(db_dir, "profiles")
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.index_path = os.path.join(db_dir, "mert_index.faiss")
        self.meta_path = os.path.join(db_dir, "meta.json")

        self._index = None
        self._id_to_song = {}  # int64 faiss id -> song_id
        self._meta = {}        # song_id -> {"raga":..., "tonic_hz":..., "title":...}
        self._load_meta()

    # ---------------------------------------------------------------- meta

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # JSON keys are always strings; cast back to int for the faiss id map.
            self._id_to_song = {int(k): v for k, v in data.get("id_to_song", {}).items()}
            self._meta = data.get("meta", {})

    def _save_meta(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"id_to_song": self._id_to_song, "meta": self._meta}, f)

    # --------------------------------------------------------------- index

    def _load_index(self, dim):
        import faiss

        if self._index is not None:
            return self._index
        if os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
        else:
            # IndexIDMap2 lets us add_with_ids/remove_ids by a stable id we
            # control, so re-indexing the same song_id replaces its row
            # instead of appending a duplicate.
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))  # cosine sim via normalized vectors
        return self._index

    def _save_index(self):
        import faiss

        faiss.write_index(self._index, self.index_path)

    # ------------------------------------------------------------- writes

    def add_song(self, song_id, music_info):
        """Index a song's global embedding for Stage-1 search and persist its
        full segment profile to disk for lazy Stage-2 fetch. Re-adding the
        same song_id replaces its previous entry rather than duplicating it."""
        if music_info.global_mert_embedding is None:
            raise ValueError(
                f"{song_id}: global_mert_embedding is None -- MERT extraction "
                "must have failed for this song (check the warnings logged by "
                "process_song/global_embeddings above). Not indexing it."
            )
        embedding = np.asarray(music_info.global_mert_embedding, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        index = self._load_index(dim=embedding.shape[0])
        faiss_id = _stable_id(song_id)

        import faiss
        try:
            index.remove_ids(np.array([faiss_id], dtype=np.int64))
        except RuntimeError:
            pass  # id wasn't present yet -- first time indexing this song

        index.add_with_ids(embedding[np.newaxis, :], np.array([faiss_id], dtype=np.int64))
        self._id_to_song[faiss_id] = song_id
        self._meta[song_id] = {
            "title": music_info.title,
            "raga": music_info.raga,
            "tonic_hz": music_info.tonic_hz,
        }
        self._save_index()
        self._save_meta()

        profile_path = os.path.join(self.profiles_dir, f"{song_id}.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(jsonpickle.encode(music_info, unpicklable=False))

        logger.info("Indexed song %s (%s)", song_id, music_info.title)

    # ------------------------------------------------------------- reads

    def search_global(self, query_embedding, top_k=20, raga_boost=0.05):
        """Stage 1: ANN search over global_mert_embedding, then a soft
        re-rank that nudges same-raga candidates up slightly (per plan.md,
        raga is a soft signal -- never a hard filter)."""
        if self._index is None:
            dim = np.asarray(query_embedding).shape[0]
            self._load_index(dim)
        if self._index.ntotal == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        search_k = min(top_k * 3, self._index.ntotal)  # over-fetch, then re-rank
        scores, indices = self._index.search(q[np.newaxis, :], search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            song_id = self._id_to_song.get(int(idx))
            if song_id is None:
                continue
            results.append({"song_id": song_id, "score": float(score),
                             "meta": self._meta.get(song_id, {})})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def load_profile(self, song_id):
        """Stage 2: lazily load one candidate song's full segment profile."""
        profile_path = os.path.join(self.profiles_dir, f"{song_id}.json")
        with open(profile_path, "r", encoding="utf-8") as f:
            return jsonpickle.decode(f.read())

    def load_profiles(self, song_ids):
        return {sid: self.load_profile(sid) for sid in song_ids}

    def list_song_ids(self):
        """All song_ids currently indexed, for DB-overview tooling
        (visualize.py's `db` mode)."""
        return sorted(self._id_to_song.values())
