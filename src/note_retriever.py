"""
RAG-based note matching module.
Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity
and temporal filtering to match context notes to flagged route-weeks.

Falls back to TF-IDF (scikit-learn) if the model can't load.
No paid API calls — everything runs locally.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from src.config import (
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    NOTE_TEMPORAL_WINDOW_DAYS,
)

import os
import logging
import warnings

# Suppress noisy third-party warnings & progress bars
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.disable(logging.WARNING)

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

# ─── Embedding backend selection & singleton model cache ─────────────────────
_USE_SBERT = False
_SHARED_MODEL = None

try:
    from sentence_transformers import SentenceTransformer
    _SHARED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    _USE_SBERT = True
except Exception:
    _USE_SBERT = False
    _SHARED_MODEL = None

if not _USE_SBERT:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim


class NoteRetriever:
    """
    Retrieves relevant context notes for a given route and time period
    using semantic similarity + temporal + route filtering.
    """

    def __init__(self, notes_df: pd.DataFrame):
        self.notes = notes_df.copy()
        self.note_texts = self.notes["note"].tolist()

        if _USE_SBERT and _SHARED_MODEL is not None:
            self.model = _SHARED_MODEL
            self.note_embeddings = self.model.encode(
                self.note_texts, normalize_embeddings=True
            )
            self._backend = "sentence-transformers"
        else:
            self._tfidf = TfidfVectorizer(stop_words="english")
            self._note_tfidf = self._tfidf.fit_transform(self.note_texts)
            self._backend = "tfidf"

    @property
    def backend(self) -> str:
        return self._backend

    def _score(self, query: str, note_idx: int) -> float:
        """Compute similarity between a query and a note."""
        if _USE_SBERT:
            q_emb = self.model.encode([query], normalize_embeddings=True)
            return float(np.dot(q_emb[0], self.note_embeddings[note_idx]))
        else:
            q_vec = self._tfidf.transform([query])
            return float(_cos_sim(q_vec, self._note_tfidf[note_idx])[0, 0])

    def find_matching_note(
        self, route: str, week_of: pd.Timestamp, cost_pct_change: float
    ) -> tuple:
        """
        Find the best matching note for a flagged route-week.

        Matching pipeline:
        1. Filter by route applicability (exact match or "All Routes")
        2. Filter by temporal proximity (within NOTE_TEMPORAL_WINDOW_DAYS)
        3. Exclude notes with negation phrases (costs not affected)
        4. Rank by semantic similarity
        5. Return best match above SIMILARITY_THRESHOLD

        Returns:
            (note_id, note_text, similarity_score) or (None, None, 0.0)
        """
        query = (
            f"Cost increase on the {route} route during the week of "
            f"{week_of.strftime('%Y-%m-%d')}. "
            f"The cost per tonne-km rose by approximately {cost_pct_change:.1f}% "
            f"compared to the route's historical average. "
            f"Looking for explanations such as weather disruptions, flooding, "
            f"fuel price changes, toll changes, festival surcharges, or other "
            f"factors that could justify higher transportation costs."
        )

        best_note_id = None
        best_note_text = None
        best_score = 0.0

        for i, (_, note_row) in enumerate(self.notes.iterrows()):
            # ── Route filter ──────────────────────────────────────────
            applies_to = note_row["applies_to"]
            if applies_to != "All Routes" and applies_to != route:
                continue

            # ── Temporal filter ───────────────────────────────────────
            note_date = note_row["date"]
            week_end = week_of + timedelta(days=6)

            if week_of <= note_date <= week_end:
                min_date_diff = 0
            else:
                min_date_diff = min(
                    abs((note_date - week_of).days),
                    abs((note_date - week_end).days),
                )

            if min_date_diff > NOTE_TEMPORAL_WINDOW_DAYS:
                continue

            # ── Negation filter ───────────────────────────────────────
            # Guard against notes that explicitly say costs weren't affected
            note_lower = note_row["note"].lower()
            cost_negation_phrases = [
                "costs were not significantly affected",
                "not significantly affected",
                "no significant disruptions",
                "costs were not affected",
                "without a rate change",
                "remained normal",
                "remained stable",
                "no major disruptions",
            ]
            if any(p in note_lower for p in cost_negation_phrases):
                continue

            # ── Similarity ────────────────────────────────────────────
            similarity = self._score(query, i)

            if similarity > best_score and similarity >= SIMILARITY_THRESHOLD:
                best_score = similarity
                best_note_id = note_row["note_id"]
                best_note_text = note_row["note"]

        if best_note_id is not None:
            return best_note_id, best_note_text, best_score
        return None, None, 0.0

    def get_nearby_notes_for_reason(
        self, route: str, week_of: pd.Timestamp
    ) -> list:
        """
        Get all notes that are temporally and route-relevant (even non-justifying).
        Used for building the 'reason' text when no justification is found.
        """
        nearby = []
        week_end = week_of + timedelta(days=6)

        for _, note_row in self.notes.iterrows():
            applies_to = note_row["applies_to"]
            if applies_to != "All Routes" and applies_to != route:
                continue

            note_date = note_row["date"]
            if week_of <= note_date <= week_end:
                date_diff = 0
            else:
                date_diff = min(
                    abs((note_date - week_of).days),
                    abs((note_date - week_end).days),
                )

            if date_diff <= NOTE_TEMPORAL_WINDOW_DAYS:
                nearby.append({
                    "note_id": note_row["note_id"],
                    "note": note_row["note"],
                    "date": note_date,
                    "date_diff": date_diff,
                })

        return nearby
